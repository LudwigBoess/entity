/**
 * @file kernels/shock_thin.hpp
 * @brief Schaal-Springel shock-surface thinning pass
 * @implements
 *   - kernel::ShockThinSurfaces_kernel<>
 * @namespaces:
 *   - kernel::
 * @note
 *   Operates on the post-detection shock-finder output. For each cell
 *   currently flagged as a shock (`Ms > 0`), it compares the local
 *   pressure-gradient magnitude `|grad p|` against the same quantity at
 *   the two neighbors along +/- n̂. If either neighbor has a strictly
 *   larger `|grad p|` and is itself flagged as a shock, the current
 *   cell's outputs are zeroed - leaving exactly one cell per shock
 *   surface as the leader along the normal direction.
 *
 *   Inputs:
 *     - `moments`: cell-centered (rho, u^i, p), used to recompute |grad p|
 *     - `pre`:     a snapshot of the shock-kernel output before thinning
 *                  (Ms in slot 0, n̂ in slots 2..4)
 *   Output:
 *     - `out`: same layout as `pre`, with non-leader cells zeroed.
 *   `out` and `pre` may NOT alias.
 */

#ifndef KERNELS_SHOCK_THIN_HPP
#define KERNELS_SHOCK_THIN_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "kernels/shock_finder.hpp"
#include "traits/metric.h"
#include "utils/error.h"
#include "utils/numeric.h"

namespace kernel {
  using namespace ntt;

  template <MetricClass M>
  class ShockThinSurfaces_kernel {
    static constexpr auto D = M::Dim;
    static_assert(M::CoordType == Coord::Cartesian,
                  "ShockThinSurfaces_kernel: only Cartesian metric supported");

    const ndfield_t<D, 5> moments;
    const ndfield_t<D, 6> pre;
    ndfield_t<D, 6>       out;

  public:
    ShockThinSurfaces_kernel(const ndfield_t<D, 5>& moments,
                             const ndfield_t<D, 6>& pre,
                             ndfield_t<D, 6>&       out)
      : moments { moments }
      , pre { pre }
      , out { out } {}

    /* per-dim: |grad p|^2 at a cell using centered differences --------- */
    Inline auto grad_p_sq(index_t i1) const -> real_t {
      const real_t g1 = INV_2 * (moments(i1 + 1, shock::prs) -
                                 moments(i1 - 1, shock::prs));
      return g1 * g1;
    }

    Inline auto grad_p_sq(index_t i1, index_t i2) const -> real_t {
      const real_t g1 = INV_2 * (moments(i1 + 1, i2, shock::prs) -
                                 moments(i1 - 1, i2, shock::prs));
      const real_t g2 = INV_2 * (moments(i1, i2 + 1, shock::prs) -
                                 moments(i1, i2 - 1, shock::prs));
      return g1 * g1 + g2 * g2;
    }

    Inline auto grad_p_sq(index_t i1, index_t i2, index_t i3) const
      -> real_t {
      const real_t g1 = INV_2 * (moments(i1 + 1, i2, i3, shock::prs) -
                                 moments(i1 - 1, i2, i3, shock::prs));
      const real_t g2 = INV_2 * (moments(i1, i2 + 1, i3, shock::prs) -
                                 moments(i1, i2 - 1, i3, shock::prs));
      const real_t g3 = INV_2 * (moments(i1, i2, i3 + 1, shock::prs) -
                                 moments(i1, i2, i3 - 1, shock::prs));
      return g1 * g1 + g2 * g2 + g3 * g3;
    }

    Inline void operator()(index_t i1) const {
      if constexpr (D == Dim::_1D) {
        const real_t Ms = pre(i1, shock::Ms_idx);
        // copy through; we'll zero below if not a leader
        out(i1, shock::Ms_idx) = Ms;
        out(i1, shock::MA_idx) = pre(i1, shock::MA_idx);
        out(i1, shock::n1_idx) = pre(i1, shock::n1_idx);
        out(i1, shock::n2_idx) = pre(i1, shock::n2_idx);
        out(i1, shock::n3_idx) = pre(i1, shock::n3_idx);
        if (Ms <= ZERO) {
          return;
        }
        const real_t n1     = pre(i1, shock::n1_idx);
        const int    di1    = static_cast<int>(SIGN(n1));
        const real_t my_gp2 = grad_p_sq(i1);
        const real_t plus_Ms  = pre(i1 + di1, shock::Ms_idx);
        const real_t minus_Ms = pre(i1 - di1, shock::Ms_idx);
        if ((plus_Ms > ZERO &&
             grad_p_sq(i1 + di1) > my_gp2) ||
            (minus_Ms > ZERO &&
             grad_p_sq(i1 - di1) > my_gp2)) {
          out(i1, shock::Ms_idx) = ZERO;
          out(i1, shock::MA_idx) = ZERO;
          out(i1, shock::n1_idx) = ZERO;
          out(i1, shock::n2_idx) = ZERO;
          out(i1, shock::n3_idx) = ZERO;
        }
      } else {
        raise::KernelError(
          HERE,
          "1D implementation of ShockThinSurfaces_kernel called for non-1D");
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim::_2D) {
        const real_t Ms = pre(i1, i2, shock::Ms_idx);
        out(i1, i2, shock::Ms_idx) = Ms;
        out(i1, i2, shock::MA_idx) = pre(i1, i2, shock::MA_idx);
        out(i1, i2, shock::n1_idx) = pre(i1, i2, shock::n1_idx);
        out(i1, i2, shock::n2_idx) = pre(i1, i2, shock::n2_idx);
        out(i1, i2, shock::n3_idx) = pre(i1, i2, shock::n3_idx);
        if (Ms <= ZERO) {
          return;
        }
        const real_t n1 = pre(i1, i2, shock::n1_idx);
        const real_t n2 = pre(i1, i2, shock::n2_idx);
        // round n̂ to the nearest cardinal/diagonal step (di1, di2 in {-1,0,1})
        const int di1 = (math::abs(n1) > static_cast<real_t>(0.25))
                          ? static_cast<int>(SIGN(n1))
                          : 0;
        const int di2 = (math::abs(n2) > static_cast<real_t>(0.25))
                          ? static_cast<int>(SIGN(n2))
                          : 0;
        if (di1 == 0 && di2 == 0) {
          return;
        }
        const real_t my_gp2 = grad_p_sq(i1, i2);
        const real_t plus_Ms = pre(i1 + di1, i2 + di2, shock::Ms_idx);
        const real_t minus_Ms = pre(i1 - di1, i2 - di2, shock::Ms_idx);
        if ((plus_Ms > ZERO &&
             grad_p_sq(i1 + di1, i2 + di2) > my_gp2) ||
            (minus_Ms > ZERO &&
             grad_p_sq(i1 - di1, i2 - di2) > my_gp2)) {
          out(i1, i2, shock::Ms_idx) = ZERO;
          out(i1, i2, shock::MA_idx) = ZERO;
          out(i1, i2, shock::n1_idx) = ZERO;
          out(i1, i2, shock::n2_idx) = ZERO;
          out(i1, i2, shock::n3_idx) = ZERO;
        }
      } else {
        raise::KernelError(
          HERE,
          "2D implementation of ShockThinSurfaces_kernel called for non-2D");
      }
    }

    Inline void operator()(index_t i1, index_t i2, index_t i3) const {
      if constexpr (D == Dim::_3D) {
        const real_t Ms = pre(i1, i2, i3, shock::Ms_idx);
        out(i1, i2, i3, shock::Ms_idx) = Ms;
        out(i1, i2, i3, shock::MA_idx) = pre(i1, i2, i3, shock::MA_idx);
        out(i1, i2, i3, shock::n1_idx) = pre(i1, i2, i3, shock::n1_idx);
        out(i1, i2, i3, shock::n2_idx) = pre(i1, i2, i3, shock::n2_idx);
        out(i1, i2, i3, shock::n3_idx) = pre(i1, i2, i3, shock::n3_idx);
        if (Ms <= ZERO) {
          return;
        }
        const real_t n1 = pre(i1, i2, i3, shock::n1_idx);
        const real_t n2 = pre(i1, i2, i3, shock::n2_idx);
        const real_t n3 = pre(i1, i2, i3, shock::n3_idx);
        const int di1 = (math::abs(n1) > static_cast<real_t>(0.25))
                          ? static_cast<int>(SIGN(n1))
                          : 0;
        const int di2 = (math::abs(n2) > static_cast<real_t>(0.25))
                          ? static_cast<int>(SIGN(n2))
                          : 0;
        const int di3 = (math::abs(n3) > static_cast<real_t>(0.25))
                          ? static_cast<int>(SIGN(n3))
                          : 0;
        if (di1 == 0 && di2 == 0 && di3 == 0) {
          return;
        }
        const real_t my_gp2 = grad_p_sq(i1, i2, i3);
        const real_t plus_Ms = pre(i1 + di1, i2 + di2, i3 + di3,
                                   shock::Ms_idx);
        const real_t minus_Ms = pre(i1 - di1, i2 - di2, i3 - di3,
                                    shock::Ms_idx);
        if ((plus_Ms > ZERO &&
             grad_p_sq(i1 + di1, i2 + di2, i3 + di3) > my_gp2) ||
            (minus_Ms > ZERO &&
             grad_p_sq(i1 - di1, i2 - di2, i3 - di3) > my_gp2)) {
          out(i1, i2, i3, shock::Ms_idx) = ZERO;
          out(i1, i2, i3, shock::MA_idx) = ZERO;
          out(i1, i2, i3, shock::n1_idx) = ZERO;
          out(i1, i2, i3, shock::n2_idx) = ZERO;
          out(i1, i2, i3, shock::n3_idx) = ZERO;
        }
      } else {
        raise::KernelError(
          HERE,
          "3D implementation of ShockThinSurfaces_kernel called for non-3D");
      }
    }
  };

} // namespace kernel

#endif // KERNELS_SHOCK_THIN_HPP
