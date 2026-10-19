/**
 * Particle exchange between domains (Metadomain::CommunicateParticles).
 *
 * Every rank builds the same deterministic particle sets for all domains:
 * alive and dead particles, and particles tagged for each communicated
 * direction, placed one cell outside the domain in that direction, with
 * payloads. After the exchange each domain must hold its own alive particles
 * unchanged and every particle sent to it, with the cell indices shifted into
 * its frame and every other quantity bitwise unchanged. When no particle
 * leaves through a periodic self-wrap, the array layout must also be the
 * one of the reference algorithm: the dead and sent slots, in tag order,
 * are refilled by the received particles in direction order, the rest is
 * appended.
 */
#include "enums.h"
#include "global.h"

#include "arch/directions.h"
#include "arch/kokkos_aliases.h"
#include "utils/error.h"

#include "metrics/minkowski.h"

#include "framework/containers/species.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

#include <Kokkos_Core.hpp>
#include <plog/Appenders/ConsoleAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Init.h>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace ntt;

namespace {

  struct Prtl {
    int      i[3] { 0, 0, 0 }, ip[3] { 0, 0, 0 };
    prtldx_t dx[3] { 0, 0, 0 }, dxp[3] { 0, 0, 0 };
    real_t   u[3] { 0, 0, 0 };
    real_t   w { 0 };
    short    tag { 0 };
    real_t   pr[2] { 0, 0 };
    npart_t  pi { 0 };
  };

  auto same(const Prtl& a, const Prtl& b, unsigned int dim) -> bool {
    for (auto d { 0u }; d < dim; ++d) {
      if (a.i[d] != b.i[d] or a.ip[d] != b.ip[d] or a.dx[d] != b.dx[d] or
          a.dxp[d] != b.dxp[d]) {
        return false;
      }
    }
    return a.u[0] == b.u[0] and a.u[1] == b.u[1] and a.u[2] == b.u[2] and
           a.w == b.w and a.tag == b.tag and a.pr[0] == b.pr[0] and
           a.pr[1] == b.pr[1] and a.pi == b.pi;
  }

  // deterministic 32-bit hash
  auto mix(std::uint32_t h) -> std::uint32_t {
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
  }

  template <Dimension D>
  struct Setup {
    using M = metric::Minkowski<D>;
    std::vector<bool>     comm_axis;
    std::vector<ncells_t> res;
  };

  /**
   * Particles of domain `idx` before the exchange: per tag a hash-chosen
   * count (0..3) of particles, shuffled; tags of directions that are not
   * communicated (towards a wall: no domain at `off` + direction on an axis
   * without periodic wrap among the `ndpd` domains) are skipped.
   */
  template <Dimension D>
  auto make_particles(unsigned int                     idx,
                      const std::vector<ncells_t>&     n,
                      const std::vector<bool>&         comm_axis,
                      const std::vector<unsigned int>& off,
                      const std::vector<unsigned int>& ndpd) -> std::vector<Prtl> {
    const auto&       dirs = dir::Directions<D>::all;
    std::vector<Prtl> ps;
    const short       ntags = static_cast<short>(2 + dirs.size());
    for (short t = 0; t < ntags; ++t) {
      if (t >= 2) {
        const auto& dr = dirs[t - 2];
        bool        ok = true;
        for (auto d { 0u }; d < static_cast<unsigned>(D); ++d) {
          const int o = static_cast<int>(off[d]) + dr[d];
          ok          = ok and (dr[d] == 0 or comm_axis[d] or
                       (o >= 0 and o < static_cast<int>(ndpd[d])));
        }
        if (not ok) {
          continue;
        }
      }
      const auto cnt = (t == 1) ? 6u : mix(idx * 977u + t * 131u + 7u) % 4u;
      for (auto k { 0u }; k < cnt; ++k) {
        Prtl p;
        p.tag = t;
        ps.push_back(p);
      }
    }
    // deterministic shuffle
    for (auto k = ps.size(); k > 1u; --k) {
      const auto j = mix(idx * 7919u + static_cast<std::uint32_t>(k)) % k;
      std::swap(ps[k - 1], ps[j]);
    }
    for (auto p { 0u }; p < ps.size(); ++p) {
      auto&         q  = ps[p];
      const npart_t id = static_cast<npart_t>(idx) * 100000u + p;
      const auto    h  = mix(static_cast<std::uint32_t>(id) + 12345u);
      q.pi             = id;
      q.pr[0]          = static_cast<real_t>(id) * static_cast<real_t>(0.5);
      q.pr[1]          = -static_cast<real_t>(h % 1000u);
      q.w = static_cast<real_t>(1 + h % 97u) * static_cast<real_t>(0.25);
      for (auto d { 0u }; d < 3u; ++d) {
        q.u[d] = static_cast<real_t>(static_cast<int>(mix(h + d) % 2001u) - 1000) *
                 static_cast<real_t>(1e-3);
      }
      for (auto d { 0u }; d < static_cast<unsigned>(D); ++d) {
        const int nd = static_cast<int>(n[d]);
        short     dr = 0;
        if (q.tag >= 2) {
          dr = dir::Directions<D>::all[q.tag - 2][d];
        }
        q.i[d]  = (dr < 0)
                    ? -1
                    : ((dr > 0) ? nd : static_cast<int>(mix(h + 11u * d) % n[d]));
        q.ip[d] = q.i[d] - dr;
        q.dx[d] = static_cast<prtldx_t>(mix(h + 3u * d) % 1024u) /
                  static_cast<prtldx_t>(1024);
        q.dxp[d] = static_cast<prtldx_t>(mix(h + 5u * d) % 1024u) /
                   static_cast<prtldx_t>(1024);
      }
    }
    return ps;
  }

  template <Dimension D>
  auto run(const std::vector<ncells_t>& res,
           const std::vector<int>&      decomposition,
           const std::vector<bool>&     comm_axis,
           const std::string&           label) -> bool {
    using M  = metric::Minkowski<D>;
    int rank = 0, nranks = 1;
#if defined(MPI_ENABLED)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
#endif
    boundaries_t<real_t> extent;
    boundaries_t<FldsBC> fldsbc;
    boundaries_t<PrtlBC> prtlbc;
    for (auto d { 0u }; d < static_cast<unsigned>(D); ++d) {
      extent.push_back({ ZERO, static_cast<real_t>(res[d]) });
      if (comm_axis[d]) {
        fldsbc.push_back({ FldsBC::PERIODIC, FldsBC::PERIODIC });
        prtlbc.push_back({ PrtlBC::PERIODIC, PrtlBC::PERIODIC });
      } else {
        fldsbc.push_back({ FldsBC::CONDUCTOR, FldsBC::CONDUCTOR });
        prtlbc.push_back({ PrtlBC::REFLECT, PrtlBC::REFLECT });
      }
    }
    const npart_t                      maxnpart = 4096;
    const std::vector<ParticleSpecies> species { ParticleSpecies(
      1u,
      "p",
      1.0f,
      1.0f,
      maxnpart,
      0u,
      1u,
      ParticlePusher::BORIS,
      false,
      RadiativeDrag::NONE,
      EmissionType::NONE,
      2u,
      1u) };
    Metadomain<SimEngine::HYBRID, M> metadomain { static_cast<unsigned int>(nranks),
                                                  decomposition,
                                                  res,
                                                  extent,
                                                  fldsbc,
                                                  prtlbc,
                                                  {},
                                                  species };
    const auto ldidx = metadomain.l_subdomain_indices()[0];
    auto&      dom   = *metadomain.subdomain_ptr(ldidx);
    auto&      sp    = dom.species[0];
    const auto dim   = static_cast<unsigned>(D);

    // the particles of every domain, and where each sent one goes
    const auto                     nd   = metadomain.ndomains();
    const auto                     ndpd = metadomain.ndomains_per_dim();
    std::vector<std::vector<Prtl>> init(nd);
    for (unsigned int idx = 0; idx < nd; ++idx) {
      init[idx] = make_particles<D>(idx,
                                    metadomain.subdomain(idx).mesh.n_active(),
                                    comm_axis,
                                    metadomain.subdomain(idx).offset_ndomains(),
                                    ndpd);
    }
    const auto domain_at = [&](std::vector<unsigned int> off) -> unsigned int {
      for (unsigned int idx = 0; idx < nd; ++idx) {
        if (metadomain.subdomain(idx).offset_ndomains() == off) {
          return idx;
        }
      }
      raise::Error("no domain at offset", HERE);
      return 0;
    };
    const auto destination = [&](unsigned int src, short tag) -> unsigned int {
      auto        off = metadomain.subdomain(src).offset_ndomains();
      const auto& dr  = dir::Directions<D>::all[tag - 2];
      for (auto d { 0u }; d < dim; ++d) {
        off[d] = static_cast<unsigned int>(
          (static_cast<int>(off[d]) + dr[d] + static_cast<int>(ndpd[d])) %
          static_cast<int>(ndpd[d]));
      }
      return domain_at(off);
    };
    // a particle as it arrives at `dst`: indices shifted into its frame
    const auto arrived = [&](const Prtl& p, unsigned int src, unsigned int dst) {
      Prtl        q  = p;
      const auto& dr = dir::Directions<D>::all[p.tag - 2];
      for (auto d { 0u }; d < dim; ++d) {
        int shift = 0;
        if (dr[d] < 0) {
          shift = static_cast<int>(metadomain.subdomain(dst).mesh.n_active()[d]);
        } else if (dr[d] > 0) {
          shift = -static_cast<int>(metadomain.subdomain(src).mesh.n_active()[d]);
        }
        q.i[d]  += shift;
        q.ip[d] += shift;
      }
      q.tag = ParticleTag::alive;
      return q;
    };

    // upload the local particles
    const auto& mine = init[ldidx];
    {
      auto i1  = Kokkos::create_mirror_view(sp.i1),
           i1p = Kokkos::create_mirror_view(sp.i1_prev);
      auto i2  = Kokkos::create_mirror_view(sp.i2),
           i2p = Kokkos::create_mirror_view(sp.i2_prev);
      auto i3  = Kokkos::create_mirror_view(sp.i3),
           i3p = Kokkos::create_mirror_view(sp.i3_prev);
      auto x1  = Kokkos::create_mirror_view(sp.dx1),
           x1p = Kokkos::create_mirror_view(sp.dx1_prev);
      auto x2  = Kokkos::create_mirror_view(sp.dx2),
           x2p = Kokkos::create_mirror_view(sp.dx2_prev);
      auto x3  = Kokkos::create_mirror_view(sp.dx3),
           x3p = Kokkos::create_mirror_view(sp.dx3_prev);
      auto u1  = Kokkos::create_mirror_view(sp.ux1),
           u2  = Kokkos::create_mirror_view(sp.ux2);
      auto u3  = Kokkos::create_mirror_view(sp.ux3),
           w   = Kokkos::create_mirror_view(sp.weight);
      auto tg  = Kokkos::create_mirror_view(sp.tag);
      auto pr  = Kokkos::create_mirror_view(sp.pld_r);
      auto pi  = Kokkos::create_mirror_view(sp.pld_i);
      for (auto p { 0u }; p < mine.size(); ++p) {
        const auto& q = mine[p];
        i1(p) = q.i[0], i1p(p) = q.ip[0], x1(p) = q.dx[0], x1p(p) = q.dxp[0];
        if constexpr (D == Dim::_2D or D == Dim::_3D) {
          i2(p) = q.i[1], i2p(p) = q.ip[1], x2(p) = q.dx[1], x2p(p) = q.dxp[1];
        }
        if constexpr (D == Dim::_3D) {
          i3(p) = q.i[2], i3p(p) = q.ip[2], x3(p) = q.dx[2], x3p(p) = q.dxp[2];
        }
        u1(p) = q.u[0], u2(p) = q.u[1], u3(p) = q.u[2], w(p) = q.w, tg(p) = q.tag;
        pr(p, 0) = q.pr[0], pr(p, 1) = q.pr[1], pi(p, 0) = q.pi;
      }
      Kokkos::deep_copy(sp.i1, i1), Kokkos::deep_copy(sp.i1_prev, i1p);
      Kokkos::deep_copy(sp.dx1, x1), Kokkos::deep_copy(sp.dx1_prev, x1p);
      if constexpr (D == Dim::_2D or D == Dim::_3D) {
        Kokkos::deep_copy(sp.i2, i2), Kokkos::deep_copy(sp.i2_prev, i2p);
        Kokkos::deep_copy(sp.dx2, x2), Kokkos::deep_copy(sp.dx2_prev, x2p);
      }
      if constexpr (D == Dim::_3D) {
        Kokkos::deep_copy(sp.i3, i3), Kokkos::deep_copy(sp.i3_prev, i3p);
        Kokkos::deep_copy(sp.dx3, x3), Kokkos::deep_copy(sp.dx3_prev, x3p);
      }
      Kokkos::deep_copy(sp.ux1, u1), Kokkos::deep_copy(sp.ux2, u2);
      Kokkos::deep_copy(sp.ux3, u3), Kokkos::deep_copy(sp.weight, w);
      Kokkos::deep_copy(sp.tag, tg), Kokkos::deep_copy(sp.pld_r, pr);
      Kokkos::deep_copy(sp.pld_i, pi);
      sp.set_npart(static_cast<npart_t>(mine.size()));
    }

    metadomain.CommunicateParticles(dom);

    // download
    std::vector<Prtl> got(sp.npart());
    {
      auto i1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {}, sp.i1);
      auto i1p = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {},
                                                     sp.i1_prev);
      auto x1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {}, sp.dx1);
      auto x1p = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {},
                                                     sp.dx1_prev);
      auto u1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {}, sp.ux1);
      auto u2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {}, sp.ux2);
      auto u3 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {}, sp.ux3);
      auto w = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {}, sp.weight);
      auto tg = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {}, sp.tag);
      auto pr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {}, sp.pld_r);
      auto pi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {}, sp.pld_i);
      decltype(i1) i2, i2p, i3, i3p;
      decltype(x1) x2, x2p, x3, x3p;
      if constexpr (D == Dim::_2D or D == Dim::_3D) {
        i2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {}, sp.i2);
        i2p = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {}, sp.i2_prev);
        x2  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {}, sp.dx2);
        x2p = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {},
                                                  sp.dx2_prev);
      }
      if constexpr (D == Dim::_3D) {
        i3 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {}, sp.i3);
        i3p = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {}, sp.i3_prev);
        x3  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {}, sp.dx3);
        x3p = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {},
                                                  sp.dx3_prev);
      }
      for (auto p { 0u }; p < got.size(); ++p) {
        auto& q = got[p];
        q.i[0] = i1(p), q.ip[0] = i1p(p), q.dx[0] = x1(p), q.dxp[0] = x1p(p);
        if constexpr (D == Dim::_2D or D == Dim::_3D) {
          q.i[1] = i2(p), q.ip[1] = i2p(p), q.dx[1] = x2(p), q.dxp[1] = x2p(p);
        }
        if constexpr (D == Dim::_3D) {
          q.i[2] = i3(p), q.ip[2] = i3p(p), q.dx[2] = x3(p), q.dxp[2] = x3p(p);
        }
        q.u[0] = u1(p), q.u[1] = u2(p), q.u[2] = u3(p), q.w = w(p), q.tag = tg(p);
        q.pr[0] = pr(p, 0), q.pr[1] = pr(p, 1), q.pi = pi(p, 0);
      }
    }

    // reference: the alive particles of this domain and the arrivals
    std::map<npart_t, Prtl> expect;
    bool                    self_wrap = false;
    for (const auto& q : mine) {
      if (q.tag == ParticleTag::alive) {
        expect[q.pi] = q;
      }
    }
    // reference layout: holes (dead, then sent in tag order), refilled by
    // the arrivals in direction order (from the domain behind each
    // direction), then appended
    std::vector<npart_t> holes;
    for (short t = 0; t < static_cast<short>(2 + dir::Directions<D>::all.size());
         ++t) {
      if (t == ParticleTag::alive) {
        continue;
      }
      for (auto p { 0u }; p < mine.size(); ++p) {
        if (mine[p].tag == t) {
          holes.push_back(p);
        }
      }
    }
    std::vector<Prtl> arrivals;
    for (short t = 2; t < static_cast<short>(2 + dir::Directions<D>::all.size());
         ++t) {
      for (unsigned int src = 0; src < nd; ++src) {
        for (const auto& q : init[src]) {
          if (q.tag == t and destination(src, t) == ldidx) {
            if (src == ldidx) {
              self_wrap = true;
            }
            const auto a = arrived(q, src, ldidx);
            arrivals.push_back(a);
            expect[a.pi] = a;
          }
        }
      }
    }
    // the direction loop visits directions in the order of Directions::all,
    // receiving from the domain behind it; within one direction the sender's
    // slot order is kept, and only one domain sends along one direction
    std::vector<Prtl> layout(mine.begin(), mine.end());
    for (auto j { 0u }; j < arrivals.size(); ++j) {
      if (j < holes.size()) {
        layout[holes[j]] = arrivals[j];
      } else {
        layout.push_back(arrivals[j]);
      }
    }
    for (auto j = arrivals.size(); j < holes.size(); ++j) {
      layout[holes[j]].tag = ParticleTag::dead;
    }

    long                   nbad_set = 0, nbad_layout = 0;
    std::map<npart_t, int> seen;
    for (const auto& q : got) {
      if (q.tag != ParticleTag::alive) {
        if (q.tag != ParticleTag::dead) {
          ++nbad_set;
        }
        continue;
      }
      const auto it = expect.find(q.pi);
      if (it == expect.end() or not same(q, it->second, dim) or seen[q.pi]++ > 0) {
        ++nbad_set;
      }
    }
    nbad_set += static_cast<long>(expect.size() - seen.size());
    if (not self_wrap) {
      if (layout.size() != got.size()) {
        ++nbad_layout;
      } else {
        for (auto p { 0u }; p < got.size(); ++p) {
          if (layout[p].tag != got[p].tag or (got[p].tag == ParticleTag::alive and
                                              not same(layout[p], got[p], dim))) {
            ++nbad_layout;
          }
        }
      }
    }
    long sw = self_wrap ? 1 : 0;
#if defined(MPI_ENABLED)
    MPI_Allreduce(MPI_IN_PLACE, &nbad_set, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &nbad_layout, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &sw, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
    const bool ok = (nbad_set == 0) and (nbad_layout == 0);
    if (rank == 0) {
      std::cout << label << " (" << nranks << " ranks): wrong or missing particles "
                << nbad_set << ", layout mismatches " << nbad_layout
                << (sw > 0 ? " (layout not checked on self-wrapping domains)" : "")
                << (ok ? "  OK" : "  FAIL") << std::endl;
    }
    return ok;
  }

} // namespace

auto main(int argc, char* argv[]) -> int {
  GlobalInitialize(argc, argv);
  // errors raised inside the framework are reported through plog
  static plog::ConsoleAppender<plog::TxtFormatter> console;
  plog::init(plog::warning, &console);
  bool ok = true;
  try {
    ok &= run<Dim::_1D>({ 40 }, { -1 }, { true }, "1D periodic");
    ok &= run<Dim::_2D>({ 24, 16 },
                        { -1, 1 },
                        { true, false },
                        "2D x1 periodic, x2 walls");
    ok &= run<Dim::_2D>({ 24, 16 }, { -1, 1 }, { true, true }, "2D periodic, x1 split");
    ok &= run<Dim::_2D>({ 16, 24 }, { 1, -1 }, { true, true }, "2D periodic, x2 split");
    ok &= run<Dim::_3D>({ 20, 12, 12 },
                        { -1, 1, 1 },
                        { true, true, true },
                        "3D periodic, x1 split");
    ok &= run<Dim::_3D>({ 12, 20, 20 },
                        { 1, -1, -1 },
                        { false, true, true },
                        "3D x1 walls, x2/x3 split");
    // walls on a split axis: one-sided corner directions next to the walls
    ok &= run<Dim::_2D>({ 24, 20 },
                        { -1, -1 },
                        { false, true },
                        "2D x1 walls, x1/x2 split");
    ok &= run<Dim::_3D>({ 24, 20, 20 },
                        { -1, -1, -1 },
                        { false, true, true },
                        "3D x1 walls, x1/x2/x3 split");
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    GlobalFinalize();
    return 1;
  }
  GlobalFinalize();
  return ok ? 0 : 1;
}
