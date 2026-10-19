/**
 * @file tests/kernels/hybrid_push.cpp
 * @brief Hybrid ion push (kernel::hybrid::Pusher_kernel, flat launch,
 *        Corrector mode) in 1D, 2D and 3D with non-zero cell-centered E, B.
 *
 * Particles sit in the active cells and up to three cells outside them, so
 * that the stencil of some reaches past the field storage. Checks:
 *   - the gather of the kernel against a host reference (per-point skip of
 *     the points outside the storage), to a relative 1e-5;
 *   - every pushed quantity and every deposited moment is finite.
 * Prints a hash of all particle arrays and moments after the push, to compare
 * builds of different versions of the kernel bit by bit.
 */

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/numeric.h"

#include "metrics/minkowski.h"

#include "kernels/hybrid/pusher.hpp"
#include "kernels/particle_shapes.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

  using namespace ntt;

  auto mix(std::uint32_t h) -> std::uint32_t {
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
  }

  // uniform in [a, b) from a hash
  auto uni(std::uint32_t h, double a, double b) -> real_t {
    return static_cast<real_t>(
      a + (b - a) * static_cast<double>(mix(h)) / 4294967296.0);
  }

  // FNV-1a over the bytes of a host view
  template <class V>
  void hash_bytes(std::uint64_t& h, const V& v) {
    const auto* b = reinterpret_cast<const unsigned char*>(v.data());
    for (std::size_t k = 0; k < v.span() * sizeof(typename V::value_type); ++k) {
      h = (h ^ b[k]) * 1099511628211ull;
    }
  }

  template <Dimension D>
  auto run(std::uint64_t& hash) -> bool {
    using metric_t = metric::Minkowski<D>;
    using kernel_t =
      kernel::hybrid::Pusher_kernel<metric_t, kernel::hybrid::PushMode::Corrector>;
    constexpr int dim = static_cast<int>(D);
    const int     n[3] { 12, (dim >= 2) ? 10 : 1, (dim >= 3) ? 8 : 1 };
    const int     G = static_cast<int>(N_GHOSTS);

    std::vector<ncells_t> res;
    boundaries_t<real_t>  ext;
    for (int d = 0; d < dim; ++d) {
      res.push_back(static_cast<ncells_t>(n[d]));
      ext.push_back({ ZERO, static_cast<real_t>(n[d]) * static_cast<real_t>(1.25) });
    }
    metric_t metric { res, ext, {} };

    // cell-centered E (0..2), B (3..5), including the ghosts
    const int e1 = n[0] + 2 * G, e2 = (dim >= 2) ? n[1] + 2 * G : 1,
              e3 = (dim >= 3) ? n[2] + 2 * G : 1;
    ndfield_t<D, 6> EB;
    if constexpr (D == Dim::_1D) {
      EB = ndfield_t<D, 6> { "EB", static_cast<ncells_t>(e1) };
    } else if constexpr (D == Dim::_2D) {
      EB = ndfield_t<D, 6> { "EB",
                             static_cast<ncells_t>(e1),
                             static_cast<ncells_t>(e2) };
    } else {
      EB = ndfield_t<D, 6> { "EB",
                             static_cast<ncells_t>(e1),
                             static_cast<ncells_t>(e2),
                             static_cast<ncells_t>(e3) };
    }
    auto       EB_h = Kokkos::create_mirror_view(EB);
    const auto eb   = [&](int c1, int c2, int c3, int c) -> real_t& {
      if constexpr (D == Dim::_1D) {
        return EB_h(c1, c);
      } else if constexpr (D == Dim::_2D) {
        return EB_h(c1, c2, c);
      } else {
        return EB_h(c1, c2, c3, c);
      }
    };
    for (int c1 = 0; c1 < e1; ++c1) {
      for (int c2 = 0; c2 < e2; ++c2) {
        for (int c3 = 0; c3 < e3; ++c3) {
          for (int c = 0; c < 6; ++c) {
            eb(c1, c2, c3, c) = uni(
              static_cast<std::uint32_t>(
                ((c3 * e2 + c2) * e1 + c1) * 6 + c + 977 * dim),
              -1.0,
              1.0);
          }
        }
      }
    }
    Kokkos::deep_copy(EB, EB_h);

    // particles: cells -3 .. n + 2 along each axis
    constexpr int  np = 600;
    ParticleArrays pa;
    pa.i1       = array_t<int*> { "i1", np };
    pa.i1_prev  = array_t<int*> { "i1_prev", np };
    pa.dx1      = array_t<prtldx_t*> { "dx1", np };
    pa.dx1_prev = array_t<prtldx_t*> { "dx1_prev", np };
    pa.i2       = array_t<int*> { "i2", np };
    pa.i2_prev  = array_t<int*> { "i2_prev", np };
    pa.dx2      = array_t<prtldx_t*> { "dx2", np };
    pa.dx2_prev = array_t<prtldx_t*> { "dx2_prev", np };
    pa.i3       = array_t<int*> { "i3", np };
    pa.i3_prev  = array_t<int*> { "i3_prev", np };
    pa.dx3      = array_t<prtldx_t*> { "dx3", np };
    pa.dx3_prev = array_t<prtldx_t*> { "dx3_prev", np };
    pa.ux1      = array_t<real_t*> { "ux1", np };
    pa.ux2      = array_t<real_t*> { "ux2", np };
    pa.ux3      = array_t<real_t*> { "ux3", np };
    pa.weight   = array_t<real_t*> { "w", np };
    pa.tag      = array_t<short*> { "tag", np };
    {
      auto i1 = Kokkos::create_mirror_view(pa.i1);
      auto i2 = Kokkos::create_mirror_view(pa.i2);
      auto i3 = Kokkos::create_mirror_view(pa.i3);
      auto x1 = Kokkos::create_mirror_view(pa.dx1);
      auto x2 = Kokkos::create_mirror_view(pa.dx2);
      auto x3 = Kokkos::create_mirror_view(pa.dx3);
      auto u1 = Kokkos::create_mirror_view(pa.ux1);
      auto u2 = Kokkos::create_mirror_view(pa.ux2);
      auto u3 = Kokkos::create_mirror_view(pa.ux3);
      auto w  = Kokkos::create_mirror_view(pa.weight);
      auto tg = Kokkos::create_mirror_view(pa.tag);
      for (int p = 0; p < np; ++p) {
        const auto h = static_cast<std::uint32_t>(p * 7 + 1000 * dim);
        i1(p)        = -3 +
                static_cast<int>(mix(h) % static_cast<std::uint32_t>(n[0] + 6));
        i2(p) = (dim >= 2)
                  ? -3 + static_cast<int>(
                           mix(h + 1) % static_cast<std::uint32_t>(n[1] + 6))
                  : 0;
        i3(p) = (dim >= 3)
                  ? -3 + static_cast<int>(
                           mix(h + 2) % static_cast<std::uint32_t>(n[2] + 6))
                  : 0;
        x1(p) = static_cast<prtldx_t>(uni(h + 3, 0.0, 1.0));
        x2(p) = static_cast<prtldx_t>(uni(h + 4, 0.0, 1.0));
        x3(p) = static_cast<prtldx_t>(uni(h + 5, 0.0, 1.0));
        u1(p) = uni(h + 6, -2.0, 2.0);
        u2(p) = uni(h + 7, -2.0, 2.0);
        u3(p) = uni(h + 8, -2.0, 2.0);
        w(p)  = uni(h + 9, 0.5, 1.5);
        tg(p) = ParticleTag::alive;
      }
      Kokkos::deep_copy(pa.i1, i1), Kokkos::deep_copy(pa.i2, i2);
      Kokkos::deep_copy(pa.i3, i3), Kokkos::deep_copy(pa.dx1, x1);
      Kokkos::deep_copy(pa.dx2, x2), Kokkos::deep_copy(pa.dx3, x3);
      Kokkos::deep_copy(pa.ux1, u1), Kokkos::deep_copy(pa.ux2, u2);
      Kokkos::deep_copy(pa.ux3, u3), Kokkos::deep_copy(pa.weight, w);
      Kokkos::deep_copy(pa.tag, tg);
    }

    const kernel::hybrid::PusherContext ctx { 1.5f, 2.0f, 0.05, 0.8, 0.7,
                                              true, n[0], n[1], n[2] };
    boundaries_t<PrtlBC>                prtl_bc;
    for (int d = 0; d < dim; ++d) {
      prtl_bc.push_back({ PrtlBC::PERIODIC, PrtlBC::PERIODIC });
    }
    const kernel::sr::PusherBoundaries<D> bc { prtl_bc };

    ndfield_t<D, 6> aux;
    if constexpr (D == Dim::_1D) {
      aux = ndfield_t<D, 6> { "aux", static_cast<ncells_t>(e1) };
    } else if constexpr (D == Dim::_2D) {
      aux = ndfield_t<D, 6> { "aux",
                              static_cast<ncells_t>(e1),
                              static_cast<ncells_t>(e2) };
    } else {
      aux = ndfield_t<D, 6> { "aux",
                              static_cast<ncells_t>(e1),
                              static_cast<ncells_t>(e2),
                              static_cast<ncells_t>(e3) };
    }

    // the gather of the kernel at the drifted sample positions
    array_t<real_t* [6]> g_kern { "g_kern", np };
    {
      auto           scatter = Kokkos::Experimental::create_scatter_view(aux);
      const kernel_t kern { ctx, bc, pa, EB, scatter, metric };
      const auto     arr = pa;
      Kokkos::parallel_for(
        "Gather",
        np,
        Lambda(prtlidx_t p) {
          vec_t<Dim::_3D> e0 { ZERO }, b0 { ZERO };
          kern.gather(arr.i1(p),
                      static_cast<real_t>(arr.dx1(p)),
                      arr.i2(p),
                      static_cast<real_t>(arr.dx2(p)),
                      arr.i3(p),
                      static_cast<real_t>(arr.dx3(p)),
                      e0,
                      b0);
          for (int c = 0; c < 3; ++c) {
            g_kern(p, c)     = e0[c];
            g_kern(p, c + 3) = b0[c];
          }
        });
    }
    // host reference
    bool ok = true;
    {
      auto gk = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {}, g_kern);
      auto i1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {}, pa.i1);
      auto i2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {}, pa.i2);
      auto i3 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {}, pa.i3);
      auto x1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {}, pa.dx1);
      auto x2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {}, pa.dx2);
      auto x3 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {}, pa.dx3);
      const int  w = kernel_t::window;
      const auto S = [](real_t d) {
        return prtl_shape::particle_shape<kernel_t::O>(math::abs(d));
      };
      int nbad = 0, nedge = 0;
      for (int p = 0; p < np; ++p) {
        double ref[6] { 0, 0, 0, 0, 0, 0 };
        bool   edge = false;
        for (int d3 = (dim >= 3 ? -w : 0); d3 <= (dim >= 3 ? w : 0); ++d3) {
          for (int d2 = (dim >= 2 ? -w : 0); d2 <= (dim >= 2 ? w : 0); ++d2) {
            for (int d1 = -w; d1 <= w; ++d1) {
              const int c1 = i1(p) + d1 + G;
              const int c2 = (dim >= 2) ? i2(p) + d2 + G : 0;
              const int c3 = (dim >= 3) ? i3(p) + d3 + G : 0;
              if (c1 < 0 or c1 >= e1 or c2 < 0 or c2 >= e2 or c3 < 0 or c3 >= e3) {
                edge = true;
                continue;
              }
              double s = S(
                static_cast<real_t>(x1(p)) - (static_cast<real_t>(d1) + HALF));
              if (dim >= 2) {
                s *= S(static_cast<real_t>(x2(p)) -
                       (static_cast<real_t>(d2) + HALF));
              }
              if (dim >= 3) {
                s *= S(static_cast<real_t>(x3(p)) -
                       (static_cast<real_t>(d3) + HALF));
              }
              for (int c = 0; c < 6; ++c) {
                ref[c] += s * static_cast<double>(eb(c1, c2, c3, c));
              }
            }
          }
        }
        nedge += edge ? 1 : 0;
        for (int c = 0; c < 6; ++c) {
          const double got = static_cast<double>(gk(p, c));
          if (not std::isfinite(got) or
              std::abs(got - ref[c]) > 1e-5 * std::max(1.0, std::abs(ref[c]))) {
            ++nbad;
          }
        }
      }
      ok = ok and (nbad == 0) and (nedge > 0);
      std::cout << dim << "D gather: " << nbad << " values off the reference, "
                << nedge << " of " << np << " particles with a clipped stencil"
                << ((nbad == 0 and nedge > 0) ? "  OK" : "  FAIL") << std::endl;
    }

    // the push itself
    {
      Kokkos::deep_copy(aux, ZERO);
      auto scatter = Kokkos::Experimental::create_scatter_view(aux);
      Kokkos::parallel_for("Push", np, kernel_t { ctx, bc, pa, EB, scatter, metric });
      Kokkos::Experimental::contribute(aux, scatter);
    }
    std::uint64_t h        = 1469598103934665603ull;
    bool          finite   = true;
    const auto    hash_and = [&](const auto& v) {
      auto vh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {}, v);
      hash_bytes(h, vh);
      if constexpr (std::is_floating_point_v<typename decltype(vh)::value_type>) {
        for (std::size_t k = 0; k < vh.span(); ++k) {
          finite = finite and std::isfinite(vh.data()[k]);
        }
      }
    };
    hash_and(pa.i1), hash_and(pa.i1_prev), hash_and(pa.dx1), hash_and(pa.dx1_prev);
    hash_and(pa.i2), hash_and(pa.i2_prev), hash_and(pa.dx2), hash_and(pa.dx2_prev);
    hash_and(pa.i3), hash_and(pa.i3_prev), hash_and(pa.dx3), hash_and(pa.dx3_prev);
    hash_and(pa.ux1), hash_and(pa.ux2), hash_and(pa.ux3), hash_and(pa.tag);
    hash_and(aux);
    ok = ok and finite;
    std::cout << dim << "D push: all values finite " << (finite ? "yes" : "NO")
              << ", hash " << std::hex << h << std::dec << std::endl;
    hash ^= h + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
    return ok;
  }

} // namespace

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  bool ok = true;
  try {
    std::uint64_t hash  = 0;
    ok                 &= run<Dim::_1D>(hash);
    ok                 &= run<Dim::_2D>(hash);
    ok                 &= run<Dim::_3D>(hash);
    std::cout << "combined hash " << std::hex << hash << std::dec << std::endl;
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return ok ? 0 : 1;
}
