/**
 * Ghost remap of the hybrid moments after a deposit: the additive halo sync,
 * the image-plasma fold at the x1 conductor walls and the ghost fills.
 *
 * 1. Every storage cell of every domain (active and ghost) holds an
 *    integer-valued deposit of comps 0..3 (V, N), so every sum is exact in
 *    single precision whatever the order. Each deposit belongs to the global
 *    active cell it maps to (periodic wrap on x2/x3, mirror about the x1
 *    walls with V_x odd). After the remap every active cell must hold the sum
 *    of the deposits that map to it, and every ghost cell the (mirrored)
 *    value of the active cell it copies. Checked for the step-by-step
 *    sequence (sync, fold, fill, wall fill) and for hybrid::RemapMoments.
 * 2. With non-integer deposits the two must agree bitwise in every cell.
 */
#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/timer.h"

#include "metrics/minkowski.h"

#include "engines/hybrid/fields_bcs.h"
#include "engines/hybrid/moments_remap.h"
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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace ntt;

namespace {

  auto hash(unsigned int idx, const int i[3], int c) -> std::uint32_t {
    std::uint32_t h = 2166136261u;
    for (const auto v : { i[0], i[1], i[2], c, static_cast<int>(idx) }) {
      h = (h ^ static_cast<std::uint32_t>(v + 1000)) * 16777619u;
    }
    return h;
  }

  // integer-valued deposit in [1, 8] of domain `idx` at storage cell `i`
  auto deposit(unsigned int idx, const int i[3], int c) -> real_t {
    return static_cast<real_t>(1u + (hash(idx, i, c) >> 9) % 8u);
  }

  // non-integer deposit in (-1, 9) (V) or (0.1, 10) (N)
  auto deposit_real(unsigned int idx, const int i[3], int c) -> real_t {
    const auto u = static_cast<double>(hash(idx, i, c)) / 4294967296.0;
    return static_cast<real_t>((c < 3) ? (10.0 * u - 1.0) : (0.1 + 9.9 * u));
  }

  // global active cell of storage cell `i` of a domain at `offset`, and the
  // sign of comp `c` under the wall mirror
  template <Dimension D>
  void image(const int                    i[3],
             const std::vector<ncells_t>& offset,
             const std::vector<ncells_t>& nglob,
             int                          g[3],
             bool&                        flip) {
    flip = false;
    for (auto d { 0u }; d < 3u; ++d) {
      g[d] = 0;
    }
    for (auto d { 0u }; d < static_cast<unsigned>(D); ++d) {
      const int n = static_cast<int>(nglob[d]);
      int x = i[d] - static_cast<int>(N_GHOSTS) + static_cast<int>(offset[d]);
      if (d == 0u) {
        if (x < 0) {
          x    = -1 - x;
          flip = true;
        } else if (x >= n) {
          x    = 2 * n - 1 - x;
          flip = true;
        }
      } else {
        x = ((x % n) + n) % n;
      }
      g[d] = x;
    }
  }

  template <Dimension D>
  using metadomain_t = Metadomain<SimEngine::HYBRID, metric::Minkowski<D>>;
  template <Dimension D>
  using domain_t = Domain<SimEngine::HYBRID, metric::Minkowski<D>>;

  // storage extents of a domain (1 for absent axes)
  template <Dimension D>
  auto extents_of(const domain_t<D>& d) -> std::vector<int> {
    std::vector<int> e { 1, 1, 1 };
    for (auto r { 0u }; r < static_cast<unsigned>(D); ++r) {
      e[r] = static_cast<int>(d.mesh.n_active()[r] + 2 * N_GHOSTS);
    }
    return e;
  }

  template <Dimension D, class V>
  auto at(const V& v, int i1, int i2, int i3, int c) -> real_t& {
    if constexpr (D == Dim::_1D) {
      (void)i2;
      (void)i3;
      return v(i1, c);
    } else if constexpr (D == Dim::_2D) {
      (void)i3;
      return v(i1, i2, c);
    } else {
      return v(i1, i2, i3, c);
    }
  }

  // aux of the local domain <- deposits (comps 0..3), zero (comps 4, 5)
  template <Dimension D>
  void fill_deposits(domain_t<D>& dom, unsigned int idx, bool integer) {
    auto       aux_h = Kokkos::create_mirror_view(dom.fields.aux);
    const auto el    = extents_of<D>(dom);
    for (int i1 = 0; i1 < el[0]; ++i1) {
      for (int i2 = 0; i2 < el[1]; ++i2) {
        for (int i3 = 0; i3 < el[2]; ++i3) {
          const int i[3] { i1, i2, i3 };
          for (int c = 0; c < 6; ++c) {
            at<D>(aux_h, i1, i2, i3, c) = (c >= 4)  ? ZERO
                                          : integer ? deposit(idx, i, c)
                                                    : deposit_real(idx, i, c);
          }
        }
      }
    }
    Kokkos::deep_copy(dom.fields.aux, aux_h);
  }

  // the remap after a deposit: the step-by-step sequence, or RemapMoments
  template <Dimension D>
  void remap(metadomain_t<D>& metadomain, domain_t<D>& dom, bool step_by_step) {
    if (step_by_step) {
      metadomain.SynchronizeFields(dom, Comm::AUX);
      hybrid::MomentsWallBC(dom, metadomain.mesh(), true);
      metadomain.CommunicateFields(dom, Comm::AUX);
      hybrid::MomentsWallBC(dom, metadomain.mesh(), false);
    } else {
      auto timers = timer::Timers { "Communications", "FieldBoundaries" };
      hybrid::RemapMoments(metadomain, dom, timers);
    }
  }

  // number of local cells that differ from the oracle, by category (active,
  // wall ghost, halo ghost, corner), and the largest difference
  template <Dimension D>
  void check_oracle(const metadomain_t<D>& metadomain,
                    domain_t<D>&           dom,
                    long                   nbad[4],
                    long                   ncell[4],
                    double&                maxdiff,
                    double&                sum_n,
                    double&                total_n) {
    auto aux_h = Kokkos::create_mirror_view(dom.fields.aux);
    Kokkos::deep_copy(aux_h, dom.fields.aux);
    const auto       nglob = metadomain.mesh().n_active();
    std::vector<int> ng { 1, 1, 1 };
    for (auto d { 0u }; d < static_cast<unsigned>(D); ++d) {
      ng[d] = static_cast<int>(nglob[d]);
    }
    const auto gidx = [&](const int g[3], int c) -> std::size_t {
      return static_cast<std::size_t>(
        ((g[2] * ng[1] + g[1]) * ng[0] + g[0]) * 4 + c);
    };
    // expected sums over the deposits of all domains, per global active cell
    std::vector<double> total(static_cast<std::size_t>(ng[0] * ng[1] * ng[2] * 4),
                              0.0);
    total_n = 0.0;
    for (unsigned int idx = 0; idx < metadomain.ndomains(); ++idx) {
      const auto& sd  = metadomain.subdomain(idx);
      const auto  e   = extents_of<D>(sd);
      const auto  off = sd.offset_ncells();
      for (int i1 = 0; i1 < e[0]; ++i1) {
        for (int i2 = 0; i2 < e[1]; ++i2) {
          for (int i3 = 0; i3 < e[2]; ++i3) {
            const int i[3] { i1, i2, i3 };
            int       g[3];
            bool      flip;
            image<D>(i, off, nglob, g, flip);
            for (int c = 0; c < 4; ++c) {
              const double v     = static_cast<double>(deposit(idx, i, c));
              total[gidx(g, c)] += (flip and c == 0) ? -v : v;
            }
            total_n += static_cast<double>(deposit(idx, i, 3));
          }
        }
      }
    }
    const auto el  = extents_of<D>(dom);
    const auto off = dom.offset_ncells();
    for (int k = 0; k < 4; ++k) {
      nbad[k] = ncell[k] = 0;
    }
    maxdiff = 0.0;
    sum_n   = 0.0;
    for (int i1 = 0; i1 < el[0]; ++i1) {
      for (int i2 = 0; i2 < el[1]; ++i2) {
        for (int i3 = 0; i3 < el[2]; ++i3) {
          const int i[3] { i1, i2, i3 };
          bool      ghost[3] { false, false, false };
          for (auto d { 0u }; d < static_cast<unsigned>(D); ++d) {
            ghost[d] = (i[d] < static_cast<int>(N_GHOSTS)) or
                       (i[d] >= el[d] - static_cast<int>(N_GHOSTS));
          }
          const bool wall_side = ghost[0] and
                                 ((i1 < static_cast<int>(N_GHOSTS) and off[0] == 0u) or
                                  (i1 >= el[0] - static_cast<int>(N_GHOSTS) and
                                   off[0] + dom.mesh.n_active()[0] == nglob[0]));
          const bool halo = (ghost[0] and not wall_side) or ghost[1] or ghost[2];
          const int cat = wall_side ? (halo ? 3 : 1) : (halo ? 2 : 0);
          int       g[3];
          bool      flip;
          image<D>(i, off, nglob, g, flip);
          bool bad = false;
          for (int c = 0; c < 4; ++c) {
            double expect = total[gidx(g, c)];
            if (flip and c == 0) {
              expect = -expect;
            }
            const real_t got  = at<D>(aux_h, i1, i2, i3, c);
            const double diff = static_cast<double>(got) - expect;
            if (diff != 0.0) {
              bad     = true;
              maxdiff = std::max(maxdiff, std::abs(diff));
            }
            if (c == 3 and cat == 0) {
              sum_n += static_cast<double>(got);
            }
          }
          ++ncell[cat];
          if (bad) {
            ++nbad[cat];
          }
        }
      }
    }
#if defined(MPI_ENABLED)
    MPI_Allreduce(MPI_IN_PLACE, nbad, 4, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, ncell, 4, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxdiff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &sum_n, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  }

  template <Dimension D>
  auto run(const std::vector<ncells_t>& res,
           const std::vector<int>&      decomposition,
           const std::string&           label) -> bool {
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
      if (d == 0u) {
        fldsbc.push_back({ FldsBC::CONDUCTOR, FldsBC::CONDUCTOR });
        prtlbc.push_back({ PrtlBC::REFLECT, PrtlBC::REFLECT });
      } else {
        fldsbc.push_back({ FldsBC::PERIODIC, FldsBC::PERIODIC });
        prtlbc.push_back({ PrtlBC::PERIODIC, PrtlBC::PERIODIC });
      }
    }
    metadomain_t<D> metadomain { static_cast<unsigned int>(nranks),
                                 decomposition,
                                 res,
                                 extent,
                                 fldsbc,
                                 prtlbc,
                                 {},
                                 {} };
    const auto      ldidx = metadomain.l_subdomain_indices()[0];
    auto&           dom   = *metadomain.subdomain_ptr(ldidx);

    bool ok = true;
    for (const bool step_by_step : { true, false }) {
      fill_deposits<D>(dom, ldidx, true);
      remap<D>(metadomain, dom, step_by_step);
      long   nbad[4], ncell[4];
      double maxdiff, sum_n, total_n;
      check_oracle<D>(metadomain, dom, nbad, ncell, maxdiff, sum_n, total_n);
      const bool good = (nbad[0] + nbad[1] + nbad[2] + nbad[3]) == 0;
      ok              = ok and good;
      if (rank == 0) {
        std::cout << label << " (" << nranks << " ranks), "
                  << (step_by_step ? "step by step" : "RemapMoments")
                  << ": mismatched cells active " << nbad[0] << "/" << ncell[0]
                  << ", wall ghost " << nbad[1] << "/" << ncell[1]
                  << ", halo ghost " << nbad[2] << "/" << ncell[2] << ", corner "
                  << nbad[3] << "/" << ncell[3] << ", max |diff| " << maxdiff
                  << "; sum N active " << sum_n << " of deposited " << total_n
                  << (good ? "  OK" : "  FAIL") << std::endl;
      }
    }

    // non-integer deposits: both remaps bitwise equal in every cell
    fill_deposits<D>(dom, ldidx, false);
    remap<D>(metadomain, dom, true);
    auto ref = Kokkos::create_mirror_view(dom.fields.aux);
    Kokkos::deep_copy(ref, dom.fields.aux);
    fill_deposits<D>(dom, ldidx, false);
    remap<D>(metadomain, dom, false);
    auto       got  = Kokkos::create_mirror_view(dom.fields.aux);
    long       ndif = 0;
    const auto el   = extents_of<D>(dom);
    Kokkos::deep_copy(got, dom.fields.aux);
    for (int i1 = 0; i1 < el[0]; ++i1) {
      for (int i2 = 0; i2 < el[1]; ++i2) {
        for (int i3 = 0; i3 < el[2]; ++i3) {
          for (int c = 0; c < 6; ++c) {
            const real_t a = at<D>(ref, i1, i2, i3, c);
            const real_t b = at<D>(got, i1, i2, i3, c);
            if (std::memcmp(&a, &b, sizeof(real_t)) != 0) {
              ++ndif;
            }
          }
        }
      }
    }
#if defined(MPI_ENABLED)
    MPI_Allreduce(MPI_IN_PLACE, &ndif, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
    ok = ok and (ndif == 0);
    if (rank == 0) {
      std::cout << label << " (" << nranks
                << " ranks), RemapMoments vs step by step, non-integer "
                   "deposits: bitwise different values "
                << ndif << ((ndif == 0) ? "  OK" : "  FAIL") << std::endl;
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
    ok &= run<Dim::_1D>({ 24 }, { -1 }, "1D x1 split");
    ok &= run<Dim::_2D>({ 16, 24 }, { 1, -1 }, "2D x2 split");
    ok &= run<Dim::_2D>({ 24, 16 }, { -1, 1 }, "2D x1 split");
    ok &= run<Dim::_3D>({ 12, 20, 20 }, { 1, -1, -1 }, "3D x2/x3 split");
    ok &= run<Dim::_3D>({ 24, 12, 12 }, { -1, 1, 1 }, "3D x1 split");
    ok &= run<Dim::_2D>({ 24, 20 }, { -1, -1 }, "2D x1/x2 split");
    ok &= run<Dim::_3D>({ 24, 20, 20 }, { -1, -1, -1 }, "3D x1/x2/x3 split");
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    GlobalFinalize();
    return 1;
  }
  GlobalFinalize();
  return ok ? 0 : 1;
}
