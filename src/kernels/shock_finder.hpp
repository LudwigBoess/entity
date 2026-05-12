/**
 * @file kernels/shock_finder.hpp
 * @brief Cell-local shock detector operating on coarse-grained particle moments
 * @implements
 *   - kernel::ShockFinder_kernel<>
 *   - enum kernel::shock::moment_idx
 *   - enum kernel::shock::out_idx
 * @namespaces:
 *   - kernel::
 *   - kernel::shock::
 * @note
 *   Output-only diagnostic. Reads cell-centered moments
 *   (rho, u^1, u^2, u^3, p) and cell-centered B^i, writes
 *   (Ms, MA, n1, n2, n3) into the first five slots of `out`.
 *   Slot 5 is reserved for a future kinetic cross-check.
 *   Non-shock cells are flagged by Ms = 0 and a zero normal vector.
 *   Currently restricted to Cartesian metrics; non-Cartesian launches
 *   should be guarded by the dispatch site.
 */

#ifndef KERNELS_SHOCK_FINDER_HPP
#define KERNELS_SHOCK_FINDER_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/metric.h"
#include "utils/error.h"
#include "utils/numeric.h"

namespace kernel {
  using namespace ntt;

  namespace shock {
    // layout of the moment input view (5 slots)
    enum moment_idx : unsigned short {
      rho = 0,
      u1  = 1,
      u2  = 2,
      u3  = 3,
      prs = 4
    };
    // layout of the output view (slots 0..4 used; 5 reserved)
    enum out_idx : unsigned short {
      Ms_idx = 0,
      MA_idx = 1,
      n1_idx = 2,
      n2_idx = 3,
      n3_idx = 4
    };
  } // namespace shock

  template <MetricClass M, bool Relativistic = false>
  class ShockFinder_kernel {
    static constexpr auto D = M::Dim;
    static_assert(M::CoordType == Coord::Cartesian,
                  "ShockFinder_kernel: only Cartesian metric supported");

    const ndfield_t<D, 5> moments;
    const ndfield_t<D, 3> bcc;
    ndfield_t<D, 6>       out;

    const real_t m_min;
    const real_t r_min;
    const int    delta;
    const real_t gamma;
    const real_t inv_b0_sq;
    const real_t v_a_floor;
    const real_t grad_p_floor;

  public:
    ShockFinder_kernel(const ndfield_t<D, 5>& moments,
                       const ndfield_t<D, 3>& bcc,
                       ndfield_t<D, 6>&       out,
                       real_t                 m_min,
                       real_t                 r_min,
                       int                    delta,
                       real_t                 gamma,
                       real_t                 inv_b0_sq,
                       real_t                 v_a_floor,
                       real_t                 grad_p_floor)
      : moments { moments }
      , bcc { bcc }
      , out { out }
      , m_min { m_min }
      , r_min { r_min }
      , delta { delta }
      , gamma { gamma }
      , inv_b0_sq { inv_b0_sq }
      , v_a_floor { v_a_floor }
      , grad_p_floor { grad_p_floor } {
      raise::ErrorIf(delta < 1,
                     "ShockFinder_kernel: delta must be >= 1",
                     HERE);
      raise::ErrorIf(gamma <= ONE,
                     "ShockFinder_kernel: gamma must be > 1",
                     HERE);
      raise::ErrorIf(static_cast<std::size_t>(delta) >= N_GHOSTS,
                     "ShockFinder_kernel: delta must be < N_GHOSTS so the "
                     "stencil fits within the synchronized halo",
                     HERE);
    }

    /* 1D linear interpolation along x1 ----------------------------------- */
    Inline auto interp_mom_1d(real_t r1, unsigned short c) const -> real_t {
      const auto i0 = static_cast<int>(math::floor(r1));
      const auto f1 = r1 - static_cast<real_t>(i0);
      return (ONE - f1) * moments(static_cast<index_t>(i0), c) +
             f1 * moments(static_cast<index_t>(i0 + 1), c);
    }

    Inline auto interp_b_1d(real_t r1, unsigned short c) const -> real_t {
      const auto i0 = static_cast<int>(math::floor(r1));
      const auto f1 = r1 - static_cast<real_t>(i0);
      return (ONE - f1) * bcc(static_cast<index_t>(i0), c) +
             f1 * bcc(static_cast<index_t>(i0 + 1), c);
    }

    /* 2D bilinear interpolation in (x1, x2) ------------------------------ */
    Inline auto interp_mom_2d(real_t         r1,
                              real_t         r2,
                              unsigned short c) const -> real_t {
      const auto i0 = static_cast<int>(math::floor(r1));
      const auto j0 = static_cast<int>(math::floor(r2));
      const auto f1 = r1 - static_cast<real_t>(i0);
      const auto f2 = r2 - static_cast<real_t>(j0);
      const auto i  = static_cast<index_t>(i0);
      const auto j  = static_cast<index_t>(j0);
      return (ONE - f1) * (ONE - f2) * moments(i, j, c) +
             f1 * (ONE - f2) * moments(i + 1, j, c) +
             (ONE - f1) * f2 * moments(i, j + 1, c) +
             f1 * f2 * moments(i + 1, j + 1, c);
    }

    Inline auto interp_b_2d(real_t         r1,
                            real_t         r2,
                            unsigned short c) const -> real_t {
      const auto i0 = static_cast<int>(math::floor(r1));
      const auto j0 = static_cast<int>(math::floor(r2));
      const auto f1 = r1 - static_cast<real_t>(i0);
      const auto f2 = r2 - static_cast<real_t>(j0);
      const auto i  = static_cast<index_t>(i0);
      const auto j  = static_cast<index_t>(j0);
      return (ONE - f1) * (ONE - f2) * bcc(i, j, c) +
             f1 * (ONE - f2) * bcc(i + 1, j, c) +
             (ONE - f1) * f2 * bcc(i, j + 1, c) +
             f1 * f2 * bcc(i + 1, j + 1, c);
    }

    /* 3D trilinear interpolation in (x1, x2, x3) ------------------------- */
    Inline auto interp_mom_3d(real_t         r1,
                              real_t         r2,
                              real_t         r3,
                              unsigned short c) const -> real_t {
      const auto i0  = static_cast<int>(math::floor(r1));
      const auto j0  = static_cast<int>(math::floor(r2));
      const auto k0  = static_cast<int>(math::floor(r3));
      const auto f1  = r1 - static_cast<real_t>(i0);
      const auto f2  = r2 - static_cast<real_t>(j0);
      const auto f3  = r3 - static_cast<real_t>(k0);
      const auto i   = static_cast<index_t>(i0);
      const auto j   = static_cast<index_t>(j0);
      const auto k   = static_cast<index_t>(k0);
      const auto c00 = (ONE - f1) * moments(i, j, k, c) +
                       f1 * moments(i + 1, j, k, c);
      const auto c10 = (ONE - f1) * moments(i, j + 1, k, c) +
                       f1 * moments(i + 1, j + 1, k, c);
      const auto c01 = (ONE - f1) * moments(i, j, k + 1, c) +
                       f1 * moments(i + 1, j, k + 1, c);
      const auto c11 = (ONE - f1) * moments(i, j + 1, k + 1, c) +
                       f1 * moments(i + 1, j + 1, k + 1, c);
      const auto c0 = (ONE - f2) * c00 + f2 * c10;
      const auto c1 = (ONE - f2) * c01 + f2 * c11;
      return (ONE - f3) * c0 + f3 * c1;
    }

    Inline auto interp_b_3d(real_t         r1,
                            real_t         r2,
                            real_t         r3,
                            unsigned short c) const -> real_t {
      const auto i0  = static_cast<int>(math::floor(r1));
      const auto j0  = static_cast<int>(math::floor(r2));
      const auto k0  = static_cast<int>(math::floor(r3));
      const auto f1  = r1 - static_cast<real_t>(i0);
      const auto f2  = r2 - static_cast<real_t>(j0);
      const auto f3  = r3 - static_cast<real_t>(k0);
      const auto i   = static_cast<index_t>(i0);
      const auto j   = static_cast<index_t>(j0);
      const auto k   = static_cast<index_t>(k0);
      const auto c00 = (ONE - f1) * bcc(i, j, k, c) +
                       f1 * bcc(i + 1, j, k, c);
      const auto c10 = (ONE - f1) * bcc(i, j + 1, k, c) +
                       f1 * bcc(i + 1, j + 1, k, c);
      const auto c01 = (ONE - f1) * bcc(i, j, k + 1, c) +
                       f1 * bcc(i + 1, j, k + 1, c);
      const auto c11 = (ONE - f1) * bcc(i, j + 1, k + 1, c) +
                       f1 * bcc(i + 1, j + 1, k + 1, c);
      const auto c0 = (ONE - f2) * c00 + f2 * c10;
      const auto c1 = (ONE - f2) * c01 + f2 * c11;
      return (ONE - f3) * c0 + f3 * c1;
    }

    /* common: Mach numbers from upstream/downstream samples -------------- */
    // Returns true if the cell satisfies the shock test, populates Ms and MA.
    //
    // Non-relativistic branch (Relativistic == false):
    //   Ms is derived from the pressure jump via the classical RH relation
    //     Ms^2 = ((gamma+1)/(2 gamma)) * (p_post/p_pre - 1) + 1
    //   This is the cheapest form and is exact for the analytic RH state in
    //   the v << c limit.
    //
    // Relativistic branch (Relativistic == true):
    //   Ms is derived from the upstream shock-frame flow speed |v_n| and
    //   the relativistic sound speed
    //     cs^2 = gamma * p / (rho + (gamma/(gamma-1)) * p)
    //   in code units (c = 1). This collapses to the Newtonian
    //   cs^2 = gamma p / rho when p << rho c^2, so the same formula works
    //   in both regimes; the relativistic branch trades the cheap
    //   pressure-ratio form for a cs-based form that stays well-behaved
    //   when p ~ rho c^2.
    Inline auto compute_machs(real_t  rho_pre,
                              real_t  rho_post,
                              real_t  p_pre,
                              real_t  p_post,
                              real_t  du_n,
                              real_t  b_pre_sq,
                              real_t& Ms,
                              real_t& MA) const -> bool {
      if (p_pre <= ZERO || rho_pre <= ZERO || rho_post <= ZERO) {
        return false;
      }
      // shock-frame upstream flow speed (RH consistency relation)
      const real_t denom = rho_post - rho_pre;
      if (denom <= ZERO) {
        return false;
      }
      const real_t v_n = du_n * rho_post / denom;
      if constexpr (Relativistic) {
        // relativistic enthalpy density and sound speed (c = 1)
        const real_t w      = rho_pre + (gamma / (gamma - ONE)) * p_pre;
        const real_t cs_sq  = gamma * p_pre / w;
        if (cs_sq <= ZERO) {
          return false;
        }
        const real_t cs = math::sqrt(cs_sq);
        Ms              = math::abs(v_n) / cs;
      } else {
        const real_t Ms_sq = ((gamma + ONE) / (TWO * gamma)) *
                               (p_post / p_pre - ONE) +
                             ONE;
        if (Ms_sq < ONE) {
          return false;
        }
        Ms = math::sqrt(Ms_sq);
      }
      if (Ms < m_min) {
        return false;
      }
      if (rho_post / rho_pre < r_min) {
        return false;
      }
      // Alfven speed (upstream); inv_b0_sq folds in the user's normalization
      const real_t v_a_sq = b_pre_sq * inv_b0_sq / rho_pre;
      const real_t v_a    = (v_a_sq > ZERO) ? math::sqrt(v_a_sq) : ZERO;
      MA = (v_a < v_a_floor) ? ZERO : math::abs(v_n) / v_a;
      return true;
    }

    Inline void operator()(index_t i1) const {
      if constexpr (D == Dim::_1D) {
        out(i1, shock::Ms_idx) = ZERO;
        out(i1, shock::MA_idx) = ZERO;
        out(i1, shock::n1_idx) = ZERO;
        out(i1, shock::n2_idx) = ZERO;
        out(i1, shock::n3_idx) = ZERO;

        const real_t div_u = INV_2 *
                             (moments(i1 + 1, shock::u1) -
                              moments(i1 - 1, shock::u1));
        if (div_u >= ZERO) {
          return;
        }

        const real_t grad_p = INV_2 * (moments(i1 + 1, shock::prs) -
                                       moments(i1 - 1, shock::prs));
        if (math::abs(grad_p) < grad_p_floor) {
          return;
        }

        const real_t n1     = -SIGN(grad_p);
        // n̂ points toward upstream (away from -∇p), so go +n̂ to reach
        // upstream and -n̂ to reach downstream.
        const real_t r_pre  = static_cast<real_t>(i1) + delta * n1;
        const real_t r_post = static_cast<real_t>(i1) - delta * n1;

        const real_t rho_pre  = interp_mom_1d(r_pre, shock::rho);
        const real_t rho_post = interp_mom_1d(r_post, shock::rho);
        const real_t p_pre    = interp_mom_1d(r_pre, shock::prs);
        const real_t p_post   = interp_mom_1d(r_post, shock::prs);
        const real_t u1_pre   = interp_mom_1d(r_pre, shock::u1);
        const real_t u1_post  = interp_mom_1d(r_post, shock::u1);
        const real_t b1_pre   = interp_b_1d(r_pre, 0);
        const real_t b2_pre   = interp_b_1d(r_pre, 1);
        const real_t b3_pre   = interp_b_1d(r_pre, 2);

        const real_t du_n     = (u1_post - u1_pre) * n1;
        const real_t b_pre_sq = NORM_SQR(b1_pre, b2_pre, b3_pre);

        real_t Ms { ZERO }, MA { ZERO };
        if (not compute_machs(rho_pre,
                              rho_post,
                              p_pre,
                              p_post,
                              du_n,
                              b_pre_sq,
                              Ms,
                              MA)) {
          return;
        }
        out(i1, shock::Ms_idx) = Ms;
        out(i1, shock::MA_idx) = MA;
        out(i1, shock::n1_idx) = n1;
      } else {
        raise::KernelError(
          HERE,
          "1D implementation of ShockFinder_kernel called for non-1D");
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim::_2D) {
        out(i1, i2, shock::Ms_idx) = ZERO;
        out(i1, i2, shock::MA_idx) = ZERO;
        out(i1, i2, shock::n1_idx) = ZERO;
        out(i1, i2, shock::n2_idx) = ZERO;
        out(i1, i2, shock::n3_idx) = ZERO;

        const real_t div_u = INV_2 * (moments(i1 + 1, i2, shock::u1) -
                                      moments(i1 - 1, i2, shock::u1) +
                                      moments(i1, i2 + 1, shock::u2) -
                                      moments(i1, i2 - 1, shock::u2));
        if (div_u >= ZERO) {
          return;
        }

        const real_t gp1 = INV_2 * (moments(i1 + 1, i2, shock::prs) -
                                    moments(i1 - 1, i2, shock::prs));
        const real_t gp2 = INV_2 * (moments(i1, i2 + 1, shock::prs) -
                                    moments(i1, i2 - 1, shock::prs));
        const real_t gp_norm = math::sqrt(gp1 * gp1 + gp2 * gp2);
        if (gp_norm < grad_p_floor) {
          return;
        }
        const real_t inv_gp = ONE / gp_norm;
        const real_t n1     = -gp1 * inv_gp;
        const real_t n2     = -gp2 * inv_gp;

        const real_t r1_pre  = static_cast<real_t>(i1) + delta * n1;
        const real_t r2_pre  = static_cast<real_t>(i2) + delta * n2;
        const real_t r1_post = static_cast<real_t>(i1) - delta * n1;
        const real_t r2_post = static_cast<real_t>(i2) - delta * n2;

        const real_t rho_pre  = interp_mom_2d(r1_pre, r2_pre, shock::rho);
        const real_t rho_post = interp_mom_2d(r1_post, r2_post, shock::rho);
        const real_t p_pre    = interp_mom_2d(r1_pre, r2_pre, shock::prs);
        const real_t p_post   = interp_mom_2d(r1_post, r2_post, shock::prs);
        const real_t u1_pre   = interp_mom_2d(r1_pre, r2_pre, shock::u1);
        const real_t u2_pre   = interp_mom_2d(r1_pre, r2_pre, shock::u2);
        const real_t u3_pre   = interp_mom_2d(r1_pre, r2_pre, shock::u3);
        const real_t u1_post  = interp_mom_2d(r1_post, r2_post, shock::u1);
        const real_t u2_post  = interp_mom_2d(r1_post, r2_post, shock::u2);
        const real_t u3_post  = interp_mom_2d(r1_post, r2_post, shock::u3);
        const real_t b1_pre   = interp_b_2d(r1_pre, r2_pre, 0);
        const real_t b2_pre   = interp_b_2d(r1_pre, r2_pre, 1);
        const real_t b3_pre   = interp_b_2d(r1_pre, r2_pre, 2);

        const real_t du_n     = (u1_post - u1_pre) * n1 +
                            (u2_post - u2_pre) * n2 +
                            (u3_post - u3_pre) * ZERO;
        const real_t b_pre_sq = NORM_SQR(b1_pre, b2_pre, b3_pre);

        real_t Ms { ZERO }, MA { ZERO };
        if (not compute_machs(rho_pre,
                              rho_post,
                              p_pre,
                              p_post,
                              du_n,
                              b_pre_sq,
                              Ms,
                              MA)) {
          return;
        }
        out(i1, i2, shock::Ms_idx) = Ms;
        out(i1, i2, shock::MA_idx) = MA;
        out(i1, i2, shock::n1_idx) = n1;
        out(i1, i2, shock::n2_idx) = n2;
      } else {
        raise::KernelError(
          HERE,
          "2D implementation of ShockFinder_kernel called for non-2D");
      }
    }

    Inline void operator()(index_t i1, index_t i2, index_t i3) const {
      if constexpr (D == Dim::_3D) {
        out(i1, i2, i3, shock::Ms_idx) = ZERO;
        out(i1, i2, i3, shock::MA_idx) = ZERO;
        out(i1, i2, i3, shock::n1_idx) = ZERO;
        out(i1, i2, i3, shock::n2_idx) = ZERO;
        out(i1, i2, i3, shock::n3_idx) = ZERO;

        const real_t div_u = INV_2 * (moments(i1 + 1, i2, i3, shock::u1) -
                                      moments(i1 - 1, i2, i3, shock::u1) +
                                      moments(i1, i2 + 1, i3, shock::u2) -
                                      moments(i1, i2 - 1, i3, shock::u2) +
                                      moments(i1, i2, i3 + 1, shock::u3) -
                                      moments(i1, i2, i3 - 1, shock::u3));
        if (div_u >= ZERO) {
          return;
        }

        const real_t gp1 = INV_2 * (moments(i1 + 1, i2, i3, shock::prs) -
                                    moments(i1 - 1, i2, i3, shock::prs));
        const real_t gp2 = INV_2 * (moments(i1, i2 + 1, i3, shock::prs) -
                                    moments(i1, i2 - 1, i3, shock::prs));
        const real_t gp3 = INV_2 * (moments(i1, i2, i3 + 1, shock::prs) -
                                    moments(i1, i2, i3 - 1, shock::prs));
        const real_t gp_norm = math::sqrt(gp1 * gp1 + gp2 * gp2 + gp3 * gp3);
        if (gp_norm < grad_p_floor) {
          return;
        }
        const real_t inv_gp = ONE / gp_norm;
        const real_t n1     = -gp1 * inv_gp;
        const real_t n2     = -gp2 * inv_gp;
        const real_t n3     = -gp3 * inv_gp;

        const real_t r1_pre  = static_cast<real_t>(i1) + delta * n1;
        const real_t r2_pre  = static_cast<real_t>(i2) + delta * n2;
        const real_t r3_pre  = static_cast<real_t>(i3) + delta * n3;
        const real_t r1_post = static_cast<real_t>(i1) - delta * n1;
        const real_t r2_post = static_cast<real_t>(i2) - delta * n2;
        const real_t r3_post = static_cast<real_t>(i3) - delta * n3;

        const real_t rho_pre = interp_mom_3d(r1_pre,
                                             r2_pre,
                                             r3_pre,
                                             shock::rho);
        const real_t rho_post = interp_mom_3d(r1_post,
                                              r2_post,
                                              r3_post,
                                              shock::rho);
        const real_t p_pre = interp_mom_3d(r1_pre, r2_pre, r3_pre, shock::prs);
        const real_t p_post = interp_mom_3d(r1_post,
                                            r2_post,
                                            r3_post,
                                            shock::prs);
        const real_t u1_pre = interp_mom_3d(r1_pre, r2_pre, r3_pre, shock::u1);
        const real_t u2_pre = interp_mom_3d(r1_pre, r2_pre, r3_pre, shock::u2);
        const real_t u3_pre = interp_mom_3d(r1_pre, r2_pre, r3_pre, shock::u3);
        const real_t u1_post = interp_mom_3d(r1_post,
                                             r2_post,
                                             r3_post,
                                             shock::u1);
        const real_t u2_post = interp_mom_3d(r1_post,
                                             r2_post,
                                             r3_post,
                                             shock::u2);
        const real_t u3_post = interp_mom_3d(r1_post,
                                             r2_post,
                                             r3_post,
                                             shock::u3);
        const real_t b1_pre = interp_b_3d(r1_pre, r2_pre, r3_pre, 0);
        const real_t b2_pre = interp_b_3d(r1_pre, r2_pre, r3_pre, 1);
        const real_t b3_pre = interp_b_3d(r1_pre, r2_pre, r3_pre, 2);

        const real_t du_n = (u1_post - u1_pre) * n1 +
                            (u2_post - u2_pre) * n2 +
                            (u3_post - u3_pre) * n3;
        const real_t b_pre_sq = NORM_SQR(b1_pre, b2_pre, b3_pre);

        real_t Ms { ZERO }, MA { ZERO };
        if (not compute_machs(rho_pre,
                              rho_post,
                              p_pre,
                              p_post,
                              du_n,
                              b_pre_sq,
                              Ms,
                              MA)) {
          return;
        }
        out(i1, i2, i3, shock::Ms_idx) = Ms;
        out(i1, i2, i3, shock::MA_idx) = MA;
        out(i1, i2, i3, shock::n1_idx) = n1;
        out(i1, i2, i3, shock::n2_idx) = n2;
        out(i1, i2, i3, shock::n3_idx) = n3;
      } else {
        raise::KernelError(
          HERE,
          "3D implementation of ShockFinder_kernel called for non-3D");
      }
    }
  };

} // namespace kernel

#endif // KERNELS_SHOCK_FINDER_HPP
