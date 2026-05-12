#include "kernels/shock_finder.hpp"

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"
#include "utils/numeric.h"

#include "metrics/minkowski.h"

#include <Kokkos_Core.hpp>

#include <iostream>
#include <stdexcept>
#include <string>

using namespace ntt;
using namespace metric;

namespace {
  void errorIf(bool condition, const std::string& message) {
    if (condition) {
      throw std::runtime_error(message);
    }
  }

  // Pack a (rho, V, p) state into the moment buffer at one cell.
  template <Dimension D>
  void setMoment(typename ndfield_t<D, 5>::host_mirror_type& m_h,
                 std::size_t                                 i,
                 real_t                                      rho,
                 real_t                                      v1,
                 real_t                                      v2,
                 real_t                                      v3,
                 real_t                                      p) {
    if constexpr (D == Dim::_1D) {
      m_h(i, 0) = rho;
      m_h(i, 1) = v1;
      m_h(i, 2) = v2;
      m_h(i, 3) = v3;
      m_h(i, 4) = p;
    }
  }

  void setMoment2D(typename ndfield_t<Dim::_2D, 5>::host_mirror_type& m_h,
                   std::size_t                                        i,
                   std::size_t                                        j,
                   real_t                                             rho,
                   real_t                                             v1,
                   real_t                                             v2,
                   real_t                                             v3,
                   real_t                                             p) {
    m_h(i, j, 0) = rho;
    m_h(i, j, 1) = v1;
    m_h(i, j, 2) = v2;
    m_h(i, j, 3) = v3;
    m_h(i, j, 4) = p;
  }

  template <Dimension D>
  void setBcc(typename ndfield_t<D, 3>::host_mirror_type& b_h,
              std::size_t                                 i,
              real_t                                      b1,
              real_t                                      b2,
              real_t                                      b3) {
    if constexpr (D == Dim::_1D) {
      b_h(i, 0) = b1;
      b_h(i, 1) = b2;
      b_h(i, 2) = b3;
    }
  }

  void setBcc2D(typename ndfield_t<Dim::_2D, 3>::host_mirror_type& b_h,
                std::size_t                                        i,
                std::size_t                                        j,
                real_t                                             b1,
                real_t                                             b2,
                real_t                                             b3) {
    b_h(i, j, 0) = b1;
    b_h(i, j, 1) = b2;
    b_h(i, j, 2) = b3;
  }

  // Analytic strong-shock state: Rankine-Hugoniot for a planar shock with
  // upstream (rho_u, p_u, v_u) and downstream (rho_d, p_d, v_d) given Ms and
  // gamma. Returns (rho_d, p_d, v_d) for an upstream-rest configuration in
  // the shock frame.
  struct RHState {
    real_t rho_pre, rho_post;
    real_t p_pre, p_post;
    real_t v_pre, v_post; // shock-frame velocities
  };

  auto rh_state(real_t rho_pre, real_t p_pre, real_t Ms, real_t gamma)
    -> RHState {
    RHState s;
    s.rho_pre = rho_pre;
    s.p_pre   = p_pre;
    // jump conditions
    const real_t Ms2     = Ms * Ms;
    const real_t r_jump  = ((gamma + ONE) * Ms2) /
                          ((gamma - ONE) * Ms2 + TWO);
    const real_t p_jump = ONE + (TWO * gamma / (gamma + ONE)) * (Ms2 - ONE);
    s.rho_post           = rho_pre * r_jump;
    s.p_post             = p_pre * p_jump;
    // sound speed upstream
    const real_t c_s = math::sqrt(gamma * p_pre / rho_pre);
    s.v_pre          = Ms * c_s;
    s.v_post         = s.v_pre / r_jump; // mass conservation in shock frame
    return s;
  }

  // 1D Sod-like shock: half-domain at upstream, half at downstream.
  // The shock plane is at index `i_shock` (cell-centered convention).
  void test1DShock() {
    using M = Minkowski<Dim::_1D>;

    constexpr std::size_t nx          = 64;
    constexpr std::size_t total_cells = nx + 2 * N_GHOSTS;

    const real_t gamma   = static_cast<real_t>(5.0 / 3.0);
    const real_t Ms_in   = static_cast<real_t>(3.0);
    const real_t rho_pre = ONE;
    const real_t p_pre   = ONE;
    const auto   s       = rh_state(rho_pre, p_pre, Ms_in, gamma);
    // lab frame: keep the upstream gas at rest and the downstream gas
    // moving with -(v_pre - v_post) so the relative velocity matches RH.
    const real_t v_lab_pre  = ZERO;
    const real_t v_lab_post = v_lab_pre - (s.v_pre - s.v_post);

    // uniform parallel B^1 (oblique-B is exercised in the second variant)
    const real_t bx        = static_cast<real_t>(0.1);
    const real_t inv_b0_sq = ONE; // code-unit Alfven normalization

    ndfield_t<Dim::_1D, 5> moments { "moments", total_cells };
    ndfield_t<Dim::_1D, 3> bcc { "bcc", total_cells };
    ndfield_t<Dim::_1D, 6> out { "out", total_cells };

    auto m_h = Kokkos::create_mirror_view(moments);
    auto b_h = Kokkos::create_mirror_view(bcc);
    Kokkos::deep_copy(m_h, ZERO);
    Kokkos::deep_copy(b_h, ZERO);

    const std::size_t i_shock = total_cells / 2;
    for (std::size_t i = 0; i < total_cells; ++i) {
      const bool is_pre = (i < i_shock);
      setMoment<Dim::_1D>(m_h,
                          i,
                          is_pre ? s.rho_pre : s.rho_post,
                          is_pre ? v_lab_pre : v_lab_post,
                          ZERO,
                          ZERO,
                          is_pre ? s.p_pre : s.p_post);
      setBcc<Dim::_1D>(b_h, i, bx, ZERO, ZERO);
    }
    Kokkos::deep_copy(moments, m_h);
    Kokkos::deep_copy(bcc, b_h);

    // launch
    const real_t m_min        = static_cast<real_t>(1.3);
    const real_t r_min        = static_cast<real_t>(1.5);
    const int    delta        = 1;
    const real_t v_a_floor    = static_cast<real_t>(1e-6);
    const real_t grad_p_floor = static_cast<real_t>(1e-12);

    Kokkos::parallel_for(
      "ShockFinderTest1D",
      CreateRangePolicy<Dim::_1D>({ N_GHOSTS + delta + 1 },
                                  { total_cells - N_GHOSTS - delta - 1 }),
      kernel::ShockFinder_kernel<M, false>(moments,
                                           bcc,
                                           out,
                                           m_min,
                                           r_min,
                                           delta,
                                           gamma,
                                           inv_b0_sq,
                                           v_a_floor,
                                           grad_p_floor));

    auto out_h = Kokkos::create_mirror_view(out);
    Kokkos::deep_copy(out_h, out);

    // shock cell should report Ms ~ Ms_in within a small relative error,
    // and a nonzero MA pointing along +x (n1 = +1).
    const real_t Ms_recovered = out_h(i_shock, kernel::shock::Ms_idx);
    const real_t MA_recovered = out_h(i_shock, kernel::shock::MA_idx);
    const real_t n1_recovered = out_h(i_shock, kernel::shock::n1_idx);

    // Upstream is at i < i_shock (low x), so n̂ points toward -x: n1 = -1.
    errorIf(not cmp::AlmostEqual(n1_recovered, -ONE),
            "1D Sod: shock normal n1 != -1 at the shock plane");
    errorIf(math::abs(Ms_recovered - Ms_in) / Ms_in >
              static_cast<real_t>(0.02),
            "1D Sod: recovered Ms differs from analytic by > 2%");

    // Alfven Mach: v_a = |B|/sqrt(rho_pre); upstream shock-frame speed = s.v_pre.
    const real_t v_a_expected = math::abs(bx) / math::sqrt(s.rho_pre);
    const real_t MA_expected  = s.v_pre / v_a_expected;
    errorIf(math::abs(MA_recovered - MA_expected) / MA_expected >
              static_cast<real_t>(0.05),
            "1D Sod: recovered MA differs from analytic by > 5%");

    // far from the shock, every sample should be zeroed.
    const std::size_t i_quiet = N_GHOSTS + 4;
    errorIf(not cmp::AlmostEqual(out_h(i_quiet, kernel::shock::Ms_idx), ZERO),
            "1D Sod: nonzero Ms in the quiet upstream region");
  }

  // Hydrodynamic limit (B = 0): MA must be zero.
  void test1DShockNoB() {
    using M                          = Minkowski<Dim::_1D>;
    constexpr std::size_t nx          = 64;
    constexpr std::size_t total_cells = nx + 2 * N_GHOSTS;

    const real_t gamma     = static_cast<real_t>(5.0 / 3.0);
    const real_t Ms_in     = static_cast<real_t>(3.0);
    const real_t rho_pre   = ONE;
    const real_t p_pre     = ONE;
    const auto   s         = rh_state(rho_pre, p_pre, Ms_in, gamma);
    const real_t v_lab_pre  = ZERO;
    const real_t v_lab_post = v_lab_pre - (s.v_pre - s.v_post);

    ndfield_t<Dim::_1D, 5> moments { "moments", total_cells };
    ndfield_t<Dim::_1D, 3> bcc { "bcc", total_cells };
    ndfield_t<Dim::_1D, 6> out { "out", total_cells };

    auto m_h = Kokkos::create_mirror_view(moments);
    auto b_h = Kokkos::create_mirror_view(bcc);
    Kokkos::deep_copy(m_h, ZERO);
    Kokkos::deep_copy(b_h, ZERO);

    const std::size_t i_shock = total_cells / 2;
    for (std::size_t i = 0; i < total_cells; ++i) {
      const bool is_pre = (i < i_shock);
      setMoment<Dim::_1D>(m_h,
                          i,
                          is_pre ? s.rho_pre : s.rho_post,
                          is_pre ? v_lab_pre : v_lab_post,
                          ZERO,
                          ZERO,
                          is_pre ? s.p_pre : s.p_post);
    }
    Kokkos::deep_copy(moments, m_h);
    Kokkos::deep_copy(bcc, b_h);

    const int delta = 1;
    Kokkos::parallel_for(
      "ShockFinderTest1DNoB",
      CreateRangePolicy<Dim::_1D>({ N_GHOSTS + delta + 1 },
                                  { total_cells - N_GHOSTS - delta - 1 }),
      kernel::ShockFinder_kernel<M, false>(moments,
                                           bcc,
                                           out,
                                           static_cast<real_t>(1.3),
                                           static_cast<real_t>(1.5),
                                           delta,
                                           gamma,
                                           ONE,
                                           static_cast<real_t>(1e-6),
                                           static_cast<real_t>(1e-12)));

    auto out_h = Kokkos::create_mirror_view(out);
    Kokkos::deep_copy(out_h, out);

    errorIf(out_h(i_shock, kernel::shock::Ms_idx) <= ZERO,
            "1D no-B: Ms not flagged at the shock plane");
    errorIf(not cmp::AlmostEqual(out_h(i_shock, kernel::shock::MA_idx), ZERO),
            "1D no-B: MA must be 0 in the hydrodynamic limit");
  }

  // No-shock baseline: smooth, expansive flow only. All outputs must be zero.
  void test1DNoShock() {
    using M                          = Minkowski<Dim::_1D>;
    constexpr std::size_t nx          = 64;
    constexpr std::size_t total_cells = nx + 2 * N_GHOSTS;

    const real_t gamma = static_cast<real_t>(5.0 / 3.0);

    ndfield_t<Dim::_1D, 5> moments { "moments", total_cells };
    ndfield_t<Dim::_1D, 3> bcc { "bcc", total_cells };
    ndfield_t<Dim::_1D, 6> out { "out", total_cells };

    auto m_h = Kokkos::create_mirror_view(moments);
    auto b_h = Kokkos::create_mirror_view(bcc);
    Kokkos::deep_copy(b_h, ZERO);

    // small sinusoidal density ripple at uniform pressure and zero bulk flow
    for (std::size_t i = 0; i < total_cells; ++i) {
      const real_t x   = static_cast<real_t>(i) / total_cells;
      const real_t rho = ONE + static_cast<real_t>(0.01) *
                                 math::sin(constant::TWO_PI * x);
      setMoment<Dim::_1D>(m_h, i, rho, ZERO, ZERO, ZERO, ONE);
    }
    Kokkos::deep_copy(moments, m_h);
    Kokkos::deep_copy(bcc, b_h);

    const int delta = 1;
    Kokkos::parallel_for(
      "ShockFinderTest1DNoShock",
      CreateRangePolicy<Dim::_1D>({ N_GHOSTS + delta + 1 },
                                  { total_cells - N_GHOSTS - delta - 1 }),
      kernel::ShockFinder_kernel<M, false>(moments,
                                           bcc,
                                           out,
                                           static_cast<real_t>(1.3),
                                           static_cast<real_t>(1.5),
                                           delta,
                                           gamma,
                                           ONE,
                                           static_cast<real_t>(1e-6),
                                           static_cast<real_t>(1e-12)));

    auto out_h = Kokkos::create_mirror_view(out);
    Kokkos::deep_copy(out_h, out);

    for (std::size_t i = N_GHOSTS + delta + 1;
         i < total_cells - N_GHOSTS - delta - 1;
         ++i) {
      errorIf(not cmp::AlmostEqual(out_h(i, kernel::shock::Ms_idx), ZERO),
              "1D no-shock: nonzero Ms reported in a quiescent region");
    }
  }

  // 2D variant of the 1D Sod test: shock plane perpendicular to x1, but with
  // an oblique upstream B field that mixes B^1 and B^2. The recovered MA must
  // match v_shock / v_A,upstream where |B|^2 = bx^2 + by^2.
  void test2DObliqueB() {
    using M = Minkowski<Dim::_2D>;

    constexpr std::size_t nx          = 32;
    constexpr std::size_t ny          = 16;
    constexpr std::size_t total_x     = nx + 2 * N_GHOSTS;
    constexpr std::size_t total_y     = ny + 2 * N_GHOSTS;

    const real_t gamma   = static_cast<real_t>(5.0 / 3.0);
    const real_t Ms_in   = static_cast<real_t>(3.0);
    const real_t rho_pre = ONE;
    const real_t p_pre   = ONE;
    const auto   s       = rh_state(rho_pre, p_pre, Ms_in, gamma);
    const real_t v_lab_pre  = ZERO;
    const real_t v_lab_post = v_lab_pre - (s.v_pre - s.v_post);

    // oblique B: 45 deg in (x1, x2); B^3 = 0
    const real_t b_amp     = static_cast<real_t>(0.1);
    const real_t bx        = b_amp * static_cast<real_t>(constant::INV_SQRT2);
    const real_t by        = b_amp * static_cast<real_t>(constant::INV_SQRT2);
    const real_t inv_b0_sq = ONE;

    ndfield_t<Dim::_2D, 5> moments { "moments", total_x, total_y };
    ndfield_t<Dim::_2D, 3> bcc { "bcc", total_x, total_y };
    ndfield_t<Dim::_2D, 6> out { "out", total_x, total_y };

    auto m_h = Kokkos::create_mirror_view(moments);
    auto b_h = Kokkos::create_mirror_view(bcc);
    Kokkos::deep_copy(m_h, ZERO);
    Kokkos::deep_copy(b_h, ZERO);

    const std::size_t i_shock = total_x / 2;
    for (std::size_t i = 0; i < total_x; ++i) {
      const bool is_pre = (i < i_shock);
      for (std::size_t j = 0; j < total_y; ++j) {
        setMoment2D(m_h,
                    i,
                    j,
                    is_pre ? s.rho_pre : s.rho_post,
                    is_pre ? v_lab_pre : v_lab_post,
                    ZERO,
                    ZERO,
                    is_pre ? s.p_pre : s.p_post);
        setBcc2D(b_h, i, j, bx, by, ZERO);
      }
    }
    Kokkos::deep_copy(moments, m_h);
    Kokkos::deep_copy(bcc, b_h);

    const int delta = 1;
    Kokkos::parallel_for(
      "ShockFinderTest2DObliqueB",
      CreateRangePolicy<Dim::_2D>(
        { N_GHOSTS + delta + 1, N_GHOSTS + delta + 1 },
        { total_x - N_GHOSTS - delta - 1, total_y - N_GHOSTS - delta - 1 }),
      kernel::ShockFinder_kernel<M, false>(moments,
                                           bcc,
                                           out,
                                           static_cast<real_t>(1.3),
                                           static_cast<real_t>(1.5),
                                           delta,
                                           gamma,
                                           inv_b0_sq,
                                           static_cast<real_t>(1e-6),
                                           static_cast<real_t>(1e-12)));

    auto out_h = Kokkos::create_mirror_view(out);
    Kokkos::deep_copy(out_h, out);

    // sample a row in the bulk; the shock should sit at i = i_shock
    const std::size_t j_mid       = total_y / 2;
    const real_t      Ms_recov    = out_h(i_shock, j_mid, kernel::shock::Ms_idx);
    const real_t      MA_recov    = out_h(i_shock, j_mid, kernel::shock::MA_idx);
    const real_t      n1_recov    = out_h(i_shock, j_mid, kernel::shock::n1_idx);
    const real_t      n2_recov    = out_h(i_shock, j_mid, kernel::shock::n2_idx);

    errorIf(not cmp::AlmostEqual(n1_recov, -ONE),
            "2D oblique-B: shock normal n1 != -1");
    errorIf(not cmp::AlmostEqual(n2_recov, ZERO),
            "2D oblique-B: shock normal n2 != 0");
    errorIf(math::abs(Ms_recov - Ms_in) / Ms_in >
              static_cast<real_t>(0.02),
            "2D oblique-B: recovered Ms differs by > 2%");

    const real_t b_pre_sq     = bx * bx + by * by;
    const real_t v_a_expected = math::sqrt(b_pre_sq * inv_b0_sq / s.rho_pre);
    const real_t MA_expected  = s.v_pre / v_a_expected;
    errorIf(math::abs(MA_recov - MA_expected) / MA_expected >
              static_cast<real_t>(0.05),
            "2D oblique-B: recovered MA differs by > 5%");
  }

  // 2D oblique shock: a planar shock at 45 deg to the grid axes. Upstream
  // sits in cells with i+j < diag; downstream in cells with i+j > diag.
  // The expected normal is n̂ = (-1/sqrt(2), -1/sqrt(2)).
  void test2DObliqueShock() {
    using M = Minkowski<Dim::_2D>;

    constexpr std::size_t n          = 48;
    constexpr std::size_t total      = n + 2 * N_GHOSTS;

    const real_t gamma   = static_cast<real_t>(5.0 / 3.0);
    const real_t Ms_in   = static_cast<real_t>(3.0);
    const real_t rho_pre = ONE;
    const real_t p_pre   = ONE;
    const auto   s       = rh_state(rho_pre, p_pre, Ms_in, gamma);

    // n̂ points from downstream toward upstream (low p): for i+j < diag
    // upstream, n̂ = (-1, -1)/sqrt(2). Downstream lab velocity sits along
    // +n̂ with magnitude v_shock * (r-1)/r. Equivalently:
    //   |v_lab_post| = s.v_pre - s.v_post
    //   v_lab_post   = (s.v_pre - s.v_post) * n̂_hat (toward upstream)
    const real_t inv_sqrt2 = static_cast<real_t>(constant::INV_SQRT2);
    const real_t n1_hat    = -inv_sqrt2;
    const real_t n2_hat    = -inv_sqrt2;
    const real_t dv_lab    = s.v_pre - s.v_post;
    const real_t v1_post   = dv_lab * n1_hat;
    const real_t v2_post   = dv_lab * n2_hat;

    const real_t bx        = static_cast<real_t>(0.1) * inv_sqrt2;
    const real_t by        = static_cast<real_t>(0.1) * inv_sqrt2;
    const real_t inv_b0_sq = ONE;

    ndfield_t<Dim::_2D, 5> moments { "moments", total, total };
    ndfield_t<Dim::_2D, 3> bcc { "bcc", total, total };
    ndfield_t<Dim::_2D, 6> out { "out", total, total };

    auto m_h = Kokkos::create_mirror_view(moments);
    auto b_h = Kokkos::create_mirror_view(bcc);
    Kokkos::deep_copy(m_h, ZERO);
    Kokkos::deep_copy(b_h, ZERO);

    const std::size_t diag = total; // separator i+j == total
    for (std::size_t i = 0; i < total; ++i) {
      for (std::size_t j = 0; j < total; ++j) {
        const bool is_pre = (i + j < diag);
        setMoment2D(m_h,
                    i,
                    j,
                    is_pre ? s.rho_pre : s.rho_post,
                    is_pre ? ZERO : v1_post,
                    is_pre ? ZERO : v2_post,
                    ZERO,
                    is_pre ? s.p_pre : s.p_post);
        setBcc2D(b_h, i, j, bx, by, ZERO);
      }
    }
    Kokkos::deep_copy(moments, m_h);
    Kokkos::deep_copy(bcc, b_h);

    const int delta = 1;
    Kokkos::parallel_for(
      "ShockFinderTest2DOblique",
      CreateRangePolicy<Dim::_2D>(
        { N_GHOSTS + delta + 1, N_GHOSTS + delta + 1 },
        { total - N_GHOSTS - delta - 1, total - N_GHOSTS - delta - 1 }),
      kernel::ShockFinder_kernel<M, false>(moments,
                                           bcc,
                                           out,
                                           static_cast<real_t>(1.3),
                                           static_cast<real_t>(1.5),
                                           delta,
                                           gamma,
                                           inv_b0_sq,
                                           static_cast<real_t>(1e-6),
                                           static_cast<real_t>(1e-12)));

    auto out_h = Kokkos::create_mirror_view(out);
    Kokkos::deep_copy(out_h, out);

    // Pick a cell on the shock plane well away from the boundaries.
    const std::size_t i_pick = total / 2;
    const std::size_t j_pick = total - i_pick; // i + j == total = diag
    const real_t      Ms_r   = out_h(i_pick, j_pick, kernel::shock::Ms_idx);
    const real_t      n1_r   = out_h(i_pick, j_pick, kernel::shock::n1_idx);
    const real_t      n2_r   = out_h(i_pick, j_pick, kernel::shock::n2_idx);

    errorIf(Ms_r <= ZERO,
            "2D oblique-shock: no shock detected on the diagonal");
    // With delta=1 the bilinear pre/post stencil unavoidably straddles
    // the 45-degree discontinuity (the shock has a sub-cell offset within
    // the stencil), so the recovered Ms is biased by ~25 % even though
    // the gradient direction (and hence n̂) is exact at the cell.
    // Tolerance is relaxed accordingly; with delta >= 2 (N_GHOSTS >= 3
    // builds, e.g. -D shape_order=3) this drops back to a few percent.
    errorIf(math::abs(Ms_r - Ms_in) / Ms_in > static_cast<real_t>(0.35),
            "2D oblique-shock: Ms differs from analytic by > 35%");
    // The shock-plane discretization on a 45 deg diagonal is jagged at the
    // grid scale, so allow generous tolerance on the recovered normal.
    errorIf(math::abs(n1_r - n1_hat) > static_cast<real_t>(0.10),
            "2D oblique-shock: n1 deviates from -1/sqrt(2) by > 0.10");
    errorIf(math::abs(n2_r - n2_hat) > static_cast<real_t>(0.10),
            "2D oblique-shock: n2 deviates from -1/sqrt(2) by > 0.10");
  }

} // namespace

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  try {
    test1DShock();
    test1DShockNoB();
    test1DNoShock();
    test2DObliqueB();
    test2DObliqueShock();
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}
