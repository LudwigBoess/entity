/**
 * @file kernels/particle_moments.hpp
 * @brief Algorithm for computing different moments from particle distribution
 * @implements
 *   - kernel::ParticleMoments_kernel<>
 *   - kernel::NormalizeVectorByRho_kernel<>
 * @namespaces:
 *   - kernel::
 * @note
 *   `ParticleMoments_kernel` has an optional `MomShOrd` template parameter
 *   (default 0). When zero, deposition uses the legacy uniform window of
 *   half-size `window`. When non-zero, the per-particle quantity is spread
 *   over a stencil of cells weighted by the particle shape function `S` of
 *   the requested order (see `kernels/particle_shapes.hpp`), so each
 *   particle contributes a partition-of-unity (SPH-like) deposition. The
 *   parameter is named `MomShOrd` (not `SHAPE_ORDER`) to avoid colliding
 *   with the compile-time `SHAPE_ORDER` macro that controls the shape
 *   used for current deposition / field interpolation in the pushers.
 */

#ifndef KERNELS_PARTICLE_MOMENTS_HPP
#define KERNELS_PARTICLE_MOMENTS_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "kernels/particle_shapes.hpp"
#include "traits/metric.h"
#include "utils/comparators.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include <vector>

namespace kernel {
  using namespace ntt;

  template <FldsID::type F>
  auto get_contrib(float mass, float charge) -> real_t {
    if constexpr (F == FldsID::Rho) {
      return mass;
    } else if constexpr (F == FldsID::Charge) {
      return charge;
    } else {
      return ONE;
    }
  }

  template <SimEngine::type S,
            MetricClass        M,
            FldsID::type       F,
            unsigned short     N,
            unsigned short     MomShOrd = 0u>
  class ParticleMoments_kernel {
    static constexpr auto D             = M::Dim;
    static constexpr bool USE_SHAPE     = (MomShOrd > 0u);
    static constexpr int  SHAPE_STENCIL = static_cast<int>(MomShOrd) + 1;

    static_assert((S != SimEngine::GRPIC) || (F != FldsID::V),
                  "Bulk velocity not supported for GRPIC");
    static_assert((F == FldsID::Rho) || (F == FldsID::Charge) || (F == FldsID::N) ||
                    (F == FldsID::Nppc) || (F == FldsID::T) || (F == FldsID::V),
                  "Invalid field ID");

    const unsigned short     c1, c2;
    scatter_ndfield_t<D, N>  Buff;
    const idx_t              buff_idx;
    const array_t<int*>      i1, i2, i3;
    const array_t<prtldx_t*> dx1, dx2, dx3;
    const array_t<real_t*>   ux1, ux2, ux3;
    const array_t<real_t*>   phi;
    const array_t<real_t*>   weight;
    const array_t<short*>    tag;
    const float              mass;
    const float              charge;
    const bool               use_weights;
    const M                  metric;
    const int                ni2;
    const unsigned short     window;

    const real_t contrib;
    const real_t smooth;
    bool         is_axis_i2min { false }, is_axis_i2max { false };

  public:
    ParticleMoments_kernel(const std::vector<unsigned short>& components,
                           const scatter_ndfield_t<D, N>&     scatter_buff,
                           idx_t                              buff_idx,
                           const array_t<int*>&               i1,
                           const array_t<int*>&               i2,
                           const array_t<int*>&               i3,
                           const array_t<prtldx_t*>&          dx1,
                           const array_t<prtldx_t*>&          dx2,
                           const array_t<prtldx_t*>&          dx3,
                           const array_t<real_t*>&            ux1,
                           const array_t<real_t*>&            ux2,
                           const array_t<real_t*>&            ux3,
                           const array_t<real_t*>&            phi,
                           const array_t<real_t*>&            weight,
                           const array_t<short*>&             tag,
                           float                              mass,
                           float                              charge,
                           bool                               use_weights,
                           const M&                           metric,
                           const boundaries_t<FldsBC>&        boundaries,
                           ncells_t                           ni2,
                           real_t                             inv_n0,
                           unsigned short                     window)
      : c1 { not components.empty() ? components[0]
                                    : static_cast<unsigned short>(0) }
      , c2 { (components.size() == 2) ? components[1]
                                      : static_cast<unsigned short>(0) }
      , Buff { scatter_buff }
      , buff_idx { buff_idx }
      , i1 { i1 }
      , i2 { i2 }
      , i3 { i3 }
      , dx1 { dx1 }
      , dx2 { dx2 }
      , dx3 { dx3 }
      , ux1 { ux1 }
      , ux2 { ux2 }
      , ux3 { ux3 }
      , phi { phi }
      , weight { weight }
      , tag { tag }
      , mass { mass }
      , charge { charge }
      , use_weights { use_weights }
      , metric { metric }
      , ni2 { static_cast<int>(ni2) }
      , window { window }
      , contrib { get_contrib<F>(mass, charge) }
      , smooth { USE_SHAPE
                   ? inv_n0
                   : inv_n0 / (real_t)(math::pow(TWO * (real_t)window + ONE,
                                                 static_cast<int>(D))) } {
      raise::ErrorIf(buff_idx >= N, "Invalid buffer index", HERE);
      raise::ErrorIf(window > N_GHOSTS, "Window size too large", HERE);
      if constexpr (USE_SHAPE) {
        // The staggered stencil reaches at most `(MomShOrd + 1) / 2` cells
        // beyond the particle's home cell on each side; that's what has
        // to fit in the ghost region (NOT the full stencil width).
        constexpr std::size_t half_reach = (MomShOrd + 1u) / 2u;
        raise::ErrorIf(half_reach > N_GHOSTS,
                       "Shape-deposition half-reach exceeds N_GHOSTS",
                       HERE);
      }
      raise::ErrorIf(((F == FldsID::Rho) || (F == FldsID::Charge)) && (mass == ZERO),
                     "Rho & Charge for massless particles not defined",
                     HERE);
      if constexpr ((M::CoordType != Coord::Cartesian) &&
                    ((D == Dim::_2D) || (D == Dim::_3D))) {
        raise::ErrorIf(boundaries.size() < 2, "boundaries defined incorrectly", HERE);
        is_axis_i2min = (boundaries[1].first == FldsBC::AXIS);
        is_axis_i2max = (boundaries[1].second == FldsBC::AXIS);
      }
    }

    Inline auto computeStressEnergyComponent(prtlidx_t p) const -> real_t {
      real_t          u0 { ZERO };
      vec_t<Dim::_3D> u_Phys { ZERO };
      if constexpr (S == SimEngine::SRPIC) {
        // stress-energy tensor for SR is computed in the tetrad (hatted) basis
        if constexpr (M::CoordType == Coord::Cartesian) {
          u_Phys[0] = ux1(p);
          u_Phys[1] = ux2(p);
          u_Phys[2] = ux3(p);
        } else {
          static_assert(D != Dim::_1D, "non-Cartesian SRPIC 1D");
          coord_t<M::PrtlDim> x_Code { ZERO };
          x_Code[0] = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p));
          x_Code[1] = static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p));
          if constexpr (D == Dim::_3D) {
            x_Code[2] = static_cast<real_t>(i3(p)) + static_cast<real_t>(dx3(p));
          } else {
            x_Code[2] = phi(p);
          }
          metric.template transform_xyz<Idx::XYZ, Idx::T>(x_Code,
                                                          { ux1(p), ux2(p), ux3(p) },
                                                          u_Phys);
        }
        u0 = (mass == ZERO)
               ? (NORM(u_Phys[0], u_Phys[1], u_Phys[2]))
               : (math::sqrt(ONE + NORM_SQR(u_Phys[0], u_Phys[1], u_Phys[2])));
      } else if constexpr (S == SimEngine::GRPIC) {
        // stress-energy tensor for GR is computed in contravariant basis
        // @TODO: proper 4D transformation needed here
        static_assert(D != Dim::_1D, "GRPIC 1D");
        coord_t<D> x_Code { ZERO };
        x_Code[0] = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p));
        x_Code[1] = static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p));
        if constexpr (D == Dim::_3D) {
          x_Code[2] = static_cast<real_t>(i3(p)) + static_cast<real_t>(dx3(p));
        }
        vec_t<Dim::_3D> u_Cntrv { ZERO };
        // compute u_i u^i for energy
        metric.template transform<Idx::D, Idx::U>(x_Code,
                                                  { ux1(p), ux2(p), ux3(p) },
                                                  u_Cntrv);
        u0 = DOT(u_Cntrv[0], u_Cntrv[1], u_Cntrv[2], ux1(p), ux2(p), ux3(p));
        u0 = (mass == ZERO) ? math::sqrt(u0) : math::sqrt(ONE + u0);
        metric.template transform<Idx::U, Idx::PU>(x_Code, u_Cntrv, u_Phys);
      } else {
        raise::KernelError(
          HERE,
          "computeStressEnergyComponent called for non-SRPIC/GRPIC");
      }
      auto T_component = (mass == ZERO ? ONE : mass) / u0;
      for (const auto& c : { c1, c2 }) {
        if (c > 0) {
          T_component *= u_Phys[c - 1];
        } else {
          T_component *= u0;
        }
      }
      return T_component;
    }

    Inline auto computeBulk3VelocityTimesMass(prtlidx_t p) const -> real_t {
      real_t          u0 { ZERO };
      // for bulk 3vel (tetrad basis)
      vec_t<Dim::_3D> u_Phys { ZERO };
      if constexpr (M::CoordType == Coord::Cartesian) {
        u_Phys[0] = ux1(p);
        u_Phys[1] = ux2(p);
        u_Phys[2] = ux3(p);
      } else {
        coord_t<M::PrtlDim> x_Code { ZERO };
        x_Code[0] = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p));
        x_Code[1] = static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p));
        if constexpr (D == Dim::_3D) {
          x_Code[2] = static_cast<real_t>(i3(p)) + static_cast<real_t>(dx3(p));
        } else {
          x_Code[2] = phi(p);
        }
        metric.template transform_xyz<Idx::XYZ, Idx::T>(x_Code,
                                                        { ux1(p), ux2(p), ux3(p) },
                                                        u_Phys);
      }
      if (mass == ZERO) {
        u0 = NORM(u_Phys[0], u_Phys[1], u_Phys[2]);
      } else {
        u0 = math::sqrt(ONE + NORM_SQR(u_Phys[0], u_Phys[1], u_Phys[2]));
      }
      // compute the corresponding moment
      return (mass == ZERO ? ONE : mass) * u_Phys[c1 - 1] / u0;
    }

    Inline void operator()(prtlidx_t p) const {
      if (tag(p) == ParticleTag::dead) {
        return;
      }
      real_t coeff { ZERO };
      if constexpr (F == FldsID::T) {
        coeff = computeStressEnergyComponent(p);
      } else if constexpr (F == FldsID::V) {
        coeff = computeBulk3VelocityTimesMass(p);
      } else {
        // for other cases, use the `contrib` defined above
        coeff = contrib;
      }
      if constexpr (F != FldsID::Nppc) {
        // for nppc calculation ...
        // ... do not take volume, weights or smoothing into account
        if constexpr (D == Dim::_1D) {
          coeff *= smooth /
                   metric.sqrt_det_h({ static_cast<real_t>(i1(p)) + HALF });
        } else if constexpr (D == Dim::_2D) {
          coeff *= smooth /
                   metric.sqrt_det_h({ static_cast<real_t>(i1(p)) + HALF,
                                       static_cast<real_t>(i2(p)) + HALF });
        } else if constexpr (D == Dim::_3D) {
          coeff *= smooth /
                   metric.sqrt_det_h({ static_cast<real_t>(i1(p)) + HALF,
                                       static_cast<real_t>(i2(p)) + HALF,
                                       static_cast<real_t>(i3(p)) + HALF });
        }
        if (use_weights) {
          coeff *= weight(p);
        }
      }
      auto buff_access = Buff.access();
      if constexpr (USE_SHAPE) {
        // Shape-function deposition: spread the per-particle quantity
        // `coeff` over a stencil of cells around the particle, weighted
        // by the product of the 1D particle shape function S evaluated
        // at each (cell_center - particle_position) offset. The shape
        // weights form a partition of unity, so total deposition is
        // exactly `coeff` per particle.
        int    i1_min { 0 };
        real_t S1[SHAPE_STENCIL];
        prtl_shape::order<true, MomShOrd>(i1(p),
                                             static_cast<real_t>(dx1(p)),
                                             i1_min,
                                             S1);
        if constexpr (D == Dim::_1D) {
          for (int n1 = 0; n1 < SHAPE_STENCIL; ++n1) {
            buff_access(i1_min + n1 + N_GHOSTS, buff_idx) += coeff * S1[n1];
          }
        } else if constexpr (D == Dim::_2D) {
          int    i2_min { 0 };
          real_t S2[SHAPE_STENCIL];
          prtl_shape::order<true, MomShOrd>(i2(p),
                                               static_cast<real_t>(dx2(p)),
                                               i2_min,
                                               S2);
          for (int n2 = 0; n2 < SHAPE_STENCIL; ++n2) {
            const int j2 = i2_min + n2;
            for (int n1 = 0; n1 < SHAPE_STENCIL; ++n1) {
              const real_t w = coeff * S1[n1] * S2[n2];
              if constexpr (M::CoordType == Coord::Cartesian) {
                buff_access(i1_min + n1 + N_GHOSTS, j2 + N_GHOSTS, buff_idx) += w;
              } else {
                // reflect contribution at axes
                if (is_axis_i2min && (j2 < 0)) {
                  buff_access(i1_min + n1 + N_GHOSTS,
                              N_GHOSTS - j2,
                              buff_idx) += w;
                } else if (is_axis_i2max && (j2 >= ni2)) {
                  buff_access(i1_min + n1 + N_GHOSTS,
                              2 * ni2 - j2 + N_GHOSTS,
                              buff_idx) += w;
                } else {
                  buff_access(i1_min + n1 + N_GHOSTS,
                              j2 + N_GHOSTS,
                              buff_idx) += w;
                }
              }
            }
          }
        } else if constexpr (D == Dim::_3D) {
          int    i2_min { 0 };
          int    i3_min { 0 };
          real_t S2[SHAPE_STENCIL];
          real_t S3[SHAPE_STENCIL];
          prtl_shape::order<true, MomShOrd>(i2(p),
                                               static_cast<real_t>(dx2(p)),
                                               i2_min,
                                               S2);
          prtl_shape::order<true, MomShOrd>(i3(p),
                                               static_cast<real_t>(dx3(p)),
                                               i3_min,
                                               S3);
          for (int n3 = 0; n3 < SHAPE_STENCIL; ++n3) {
            for (int n2 = 0; n2 < SHAPE_STENCIL; ++n2) {
              const int j2 = i2_min + n2;
              for (int n1 = 0; n1 < SHAPE_STENCIL; ++n1) {
                const real_t w = coeff * S1[n1] * S2[n2] * S3[n3];
                if constexpr (M::CoordType == Coord::Cartesian) {
                  buff_access(i1_min + n1 + N_GHOSTS,
                              j2 + N_GHOSTS,
                              i3_min + n3 + N_GHOSTS,
                              buff_idx) += w;
                } else {
                  if (is_axis_i2min && (j2 < 0)) {
                    buff_access(i1_min + n1 + N_GHOSTS,
                                N_GHOSTS - j2,
                                i3_min + n3 + N_GHOSTS,
                                buff_idx) += w;
                  } else if (is_axis_i2max && (j2 >= ni2)) {
                    buff_access(i1_min + n1 + N_GHOSTS,
                                2 * ni2 - j2 + N_GHOSTS,
                                i3_min + n3 + N_GHOSTS,
                                buff_idx) += w;
                  } else {
                    buff_access(i1_min + n1 + N_GHOSTS,
                                j2 + N_GHOSTS,
                                i3_min + n3 + N_GHOSTS,
                                buff_idx) += w;
                  }
                }
              }
            }
          }
        }
      } else if constexpr (D == Dim::_1D) {
        for (auto di1 { -window }; di1 <= window; ++di1) {
          buff_access(i1(p) + di1 + N_GHOSTS, buff_idx) += coeff;
        }
      } else if constexpr (D == Dim::_2D) {
        for (auto di2 { -window }; di2 <= window; ++di2) {
          for (auto di1 { -window }; di1 <= window; ++di1) {
            if constexpr (M::CoordType == Coord::Cartesian) {
              buff_access(i1(p) + di1 + N_GHOSTS,
                          i2(p) + di2 + N_GHOSTS,
                          buff_idx) += coeff;
            } else {
              // reflect contribution at axes
              if (is_axis_i2min && (i2(p) + di2 < 0)) {
                buff_access(i1(p) + di1 + N_GHOSTS,
                            N_GHOSTS - (i2(p) + di2),
                            buff_idx) += coeff;
              } else if (is_axis_i2max && (i2(p) + di2 >= ni2)) {
                buff_access(i1(p) + di1 + N_GHOSTS,
                            2 * ni2 - (i2(p) + di2) + N_GHOSTS,
                            buff_idx) += coeff;
              } else {
                buff_access(i1(p) + di1 + N_GHOSTS,
                            i2(p) + di2 + N_GHOSTS,
                            buff_idx) += coeff;
              }
            }
          }
        }
      } else if constexpr (D == Dim::_3D) {
        for (auto di3 { -window }; di3 <= window; ++di3) {
          for (auto di2 { -window }; di2 <= window; ++di2) {
            for (auto di1 { -window }; di1 <= window; ++di1) {
              if constexpr (M::CoordType == Coord::Cartesian) {
                buff_access(i1(p) + di1 + N_GHOSTS,
                            i2(p) + di2 + N_GHOSTS,
                            i3(p) + di3 + N_GHOSTS,
                            buff_idx) += coeff;
              } else {
                // reflect contribution at axes
                if (is_axis_i2min && (i2(p) + di2 < 0)) {
                  buff_access(i1(p) + di1 + N_GHOSTS,
                              N_GHOSTS - (i2(p) + di2),
                              i3(p) + di3 + N_GHOSTS,
                              buff_idx) += coeff;
                } else if (is_axis_i2max && (i2(p) + di2 >= ni2)) {
                  buff_access(i1(p) + di1 + N_GHOSTS,
                              2 * ni2 - (i2(p) + di2) + N_GHOSTS,
                              i3(p) + di3 + N_GHOSTS,
                              buff_idx) += coeff;
                } else {
                  buff_access(i1(p) + di1 + N_GHOSTS,
                              i2(p) + di2 + N_GHOSTS,
                              i3(p) + di3 + N_GHOSTS,
                              buff_idx) += coeff;
                }
              }
            }
          }
        }
      }
    }
  };

  template <Dimension D, unsigned short N>
  class NormalizeVectorByRho_kernel {
    const ndfield_t<D, N> Rho;
    ndfield_t<D, N>       Vector;
    const unsigned short  c_rho, c_v1, c_v2, c_v3;

  public:
    NormalizeVectorByRho_kernel(const ndfield_t<D, N>& rho,
                                const ndfield_t<D, N>& vector,
                                unsigned short         crho,
                                unsigned short         cv1,
                                unsigned short         cv2,
                                unsigned short         cv3)
      : Rho { rho }
      , Vector { vector }
      , c_rho { crho }
      , c_v1 { cv1 }
      , c_v2 { cv2 }
      , c_v3 { cv3 } {
      raise::ErrorIf(c_rho >= N or c_v1 >= N or c_v2 >= N or c_v3 >= N,
                     "Invalid component index",
                     HERE);
      raise::ErrorIf(c_rho == c_v1 or c_rho == c_v2 or c_rho == c_v3,
                     "Invalid component index",
                     HERE);
      raise::ErrorIf(c_v1 == c_v2 or c_v1 == c_v3 or c_v2 == c_v3,
                     "Invalid component index",
                     HERE);
    }

    Inline void operator()(cellidx_t i1) const {
      if constexpr (D == Dim::_1D) {
        if (not cmp::AlmostZero(Rho(i1, c_rho))) {
          Vector(i1, c_v1) /= Rho(i1, c_rho);
          Vector(i1, c_v2) /= Rho(i1, c_rho);
          Vector(i1, c_v3) /= Rho(i1, c_rho);
        }
      } else {
        raise::KernelError(
          HERE,
          "1D implementation of NormalizeVectorByRho_kernel called for non-1D");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2) const {
      if constexpr (D == Dim::_2D) {
        if (not cmp::AlmostZero(Rho(i1, i2, c_rho))) {
          Vector(i1, i2, c_v1) /= Rho(i1, i2, c_rho);
          Vector(i1, i2, c_v2) /= Rho(i1, i2, c_rho);
          Vector(i1, i2, c_v3) /= Rho(i1, i2, c_rho);
        }
      } else {
        raise::KernelError(
          HERE,
          "2D implementation of NormalizeVectorByRho_kernel called for non-2D");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2, cellidx_t i3) const {
      if constexpr (D == Dim::_3D) {
        if (not cmp::AlmostZero(Rho(i1, i2, i3, c_rho))) {
          Vector(i1, i2, i3, c_v1) /= Rho(i1, i2, i3, c_rho);
          Vector(i1, i2, i3, c_v2) /= Rho(i1, i2, i3, c_rho);
          Vector(i1, i2, i3, c_v3) /= Rho(i1, i2, i3, c_rho);
        }
      } else {
        raise::KernelError(
          HERE,
          "3D implementation of NormalizeVectorByRho_kernel called for non-3D");
      }
    }
  };

} // namespace kernel

#endif // KERNELS_PARTICLE_MOMENTS_HPP
