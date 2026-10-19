/**
 * @file engines/hybrid/fields_bcs.h
 * @brief Non-periodic (reflecting-wall / perfect-conductor) boundary
 *        conditions for the HYBRID engine, x1 walls only.
 * @implements
 *   - ntt::hybrid::FieldBoundaries<>          -> void   (em: Ee and/or Bf)
 *   - ntt::hybrid::WallEPrime<>               -> void   (em0::345 = Ee'/Ee'')
 *   - ntt::hybrid::WallScratchB<>             -> void   (cur = Bf*, Bf**)
 *   - ntt::hybrid::WallBckpFill<>             -> void   (bckp = Ec'/Bc')
 *   - ntt::hybrid::MomentsWallFill<>          -> void   (any moments buffer)
 *   - ntt::hybrid::MomentsWallBC<>            -> void   (aux = V, N)
 * @namespaces:
 *   - ntt::hybrid::
 *
 * PERIODIC boundaries are realized by the metadomain halo exchange
 * (Metadomain::CommunicateFields) and are skipped here.
 *
 * The conductor wall (kernels/hybrid/wall_bcs.hpp) enforces E_tan = 0 on the
 * wall plane; Faraday then freezes the wall-plane B_n automatically, which is
 * the correct condition for an OBLIQUE background field. The wall-plane B_n is
 * deliberately never overwritten (zeroing it is only valid for perpendicular
 * fields and plants a div(B) layer at the wall). B ghosts get an even
 * (zero-gradient) mirror so the Hall term sees no fake wall current.
 *
 * EVERY buffer whose wall ghosts (or wall-plane values) enter a stencil needs
 * explicit treatment -- there is no halo exchange at a physical wall, so
 * untreated ghosts keep their allocation value (zero) forever:
 *   - em  (Ee/Bf)         -> FieldBoundaries (BC::E and/or BC::B)
 *   - em0::345 (Ee'/Ee'') -> WallEPrime, before Faraday #2/#3
 *   - cur (Bf*, Bf**)     -> WallScratchB, before EMF #1/#2 (Hall stencil)
 *   - bckp (Ec'/Bc')      -> WallBckpFill, before the particle gathers
 *   - aux (V/N)           -> MomentsWallBC after each deposit: folds the
 *     ghost deposit tails back as the image-plasma contribution (V_x odd) and
 *     mirror-fills the ghosts for the EMF / filter stencils
 *
 * @note Hybrid is Cartesian Minkowski, so only the conductor wall is
 *       meaningful (no axis / GR-horizon boundaries).
 * @note Only x1 walls are implemented (the hybrid shock geometry); a
 *       CONDUCTOR boundary in x2/x3 raises.
 */

#ifndef ENGINES_HYBRID_FIELDS_BCS_H
#define ENGINES_HYBRID_FIELDS_BCS_H

#include "enums.h"
#include "global.h"

#include "arch/directions.h"
#include "arch/kokkos_aliases.h"
#include "utils/error.h"

#include "metrics/minkowski.h"

#include "framework/domain/domain.h"
#include "framework/domain/grid.h"
#include "kernels/hybrid/wall_bcs.hpp"

#include <Kokkos_Core.hpp>

#include <type_traits>
#include <vector>

namespace ntt {
  namespace hybrid {

    namespace wall {

      /**
       * @brief Range over `rows` wall-normal rows (i1 = 0 .. rows) of the wall
       *        layer, over the full transverse extent, or the active one when
       *        `active_transverse`.
       */
      template <Dimension D>
      auto WallRange(Domain<SimEngine::HYBRID, metric::Minkowski<D>>& domain,
                     ncells_t                                         rows,
                     bool active_transverse) -> range_t<D> {
        std::vector<ncells_t> xi_min, xi_max;
        xi_min.push_back(0);
        xi_max.push_back(rows);
        const std::vector<in> all_dirs { in::x1, in::x2, in::x3 };
        for (auto d { 1u }; d < static_cast<unsigned>(D); ++d) {
          const auto dd = all_dirs[d];
          if (active_transverse) {
            xi_min.push_back(domain.mesh.i_min(dd));
            xi_max.push_back(domain.mesh.i_max(dd));
          } else {
            xi_min.push_back(0);
            xi_max.push_back(domain.mesh.n_all(dd));
          }
        }
        if constexpr (D == Dim::_1D) {
          return CreateRangePolicy<D>({ xi_min[0] }, { xi_max[0] });
        } else if constexpr (D == Dim::_2D) {
          return CreateRangePolicy<D>({ xi_min[0], xi_min[1] },
                                      { xi_max[0], xi_max[1] });
        } else {
          return CreateRangePolicy<D>({ xi_min[0], xi_min[1], xi_min[2] },
                                      { xi_max[0], xi_max[1], xi_max[2] });
        }
      }

      // wall-normal rows of the x1-staggered (edge-E / face-B) kernels: the
      // wall plane i1 = 0 and the ghost layers (the +x1 side has one node less)
      inline constexpr ncells_t staggered_rows_lo = N_GHOSTS + 1;
      inline constexpr ncells_t staggered_rows_hi = N_GHOSTS;

      /**
       * @brief The kernel of the -x1 wall on rows [0, nlo) of a launch and the
       *        kernel of the +x1 wall on the rows after them (row = the first
       *        index, the wall-normal layer): one launch for both walls. The
       *        two walls touch disjoint cells, so the result is that of two
       *        launches.
       */
      template <class KLo, class KHi>
      struct BothWalls_kernel {
        KLo       lo;
        KHi       hi;
        cellidx_t nlo;

        Inline void operator()(cellidx_t i1) const {
          if (i1 < nlo) {
            lo(i1);
          } else {
            hi(i1 - nlo);
          }
        }

        Inline void operator()(cellidx_t i1, cellidx_t i2) const {
          if (i1 < nlo) {
            lo(i1, i2);
          } else {
            hi(i1 - nlo, i2);
          }
        }

        Inline void operator()(cellidx_t i1, cellidx_t i2, cellidx_t i3) const {
          if (i1 < nlo) {
            lo(i1, i2, i3);
          } else {
            hi(i1 - nlo, i2, i3);
          }
        }
      };

      /**
       * @brief Invoke `f(sign, i_edge)` for every x1 conductor wall this rank
       *        owns. Raises on non-x1 conductor walls (unimplemented).
       */
      template <Dimension D, typename F>
      void ForEachX1Wall(Domain<SimEngine::HYBRID, metric::Minkowski<D>>& domain,
                         const Grid<D>& global_grid,
                         F              f) {
        for (const auto& direction : dir::Directions<D>::orth) {
          const auto gbc = global_grid.flds_bc_in(direction);
          if (gbc == FldsBC::PERIODIC) {
            continue; // realized by the halo exchange
          } else if (gbc == FldsBC::CONDUCTOR) {
            if (domain.mesh.flds_bc_in(direction) != FldsBC::CONDUCTOR) {
              continue; // not an edge-owning rank
            }
            const auto dim  = direction.get_dim();
            const auto sign = direction.get_sign();
            raise::ErrorIf(dim != in::x1,
                           "hybrid engine: CONDUCTOR walls are only "
                           "implemented in x1",
                           HERE);
            const ncells_t i_edge = (sign > 0) ? domain.mesh.i_max(in::x1)
                                               : domain.mesh.i_min(in::x1);
            f(sign, i_edge);
          } else {
            raise::Error("hybrid engine: only PERIODIC and CONDUCTOR field "
                         "boundaries are implemented",
                         HERE);
          }
        }
      }

      /**
       * @brief Launch the wall kernels `make(std::false_type {}, i_edge)`
       *        (-x1 wall, `rows_lo` wall-normal rows) and
       *        `make(std::true_type {}, i_edge)` (+x1 wall, `rows_hi` rows) of
       *        the x1 conductor walls this rank owns, in one launch when it
       *        owns both; transverse extent as in WallRange.
       */
      template <Dimension D, class Make>
      void LaunchX1Walls(const char* name,
                         Domain<SimEngine::HYBRID, metric::Minkowski<D>>& domain,
                         const Grid<D>& global_grid,
                         ncells_t       rows_lo,
                         ncells_t       rows_hi,
                         bool           active_transverse,
                         Make           make) {
        bool has_lo = false, has_hi = false;
        ForEachX1Wall<D>(domain, global_grid, [&](short sign, ncells_t) {
          if (sign < 0) {
            has_lo = true;
          } else {
            has_hi = true;
          }
        });
        const ncells_t i_lo = domain.mesh.i_min(in::x1);
        const ncells_t i_hi = domain.mesh.i_max(in::x1);
        if (has_lo and has_hi) {
          // each wall reads and writes N_GHOSTS cells on either side of its
          // plane: the two are disjoint in a domain wider than 2 N_GHOSTS
          raise::ErrorIf(i_hi - i_lo <= 2 * N_GHOSTS,
                         "hybrid walls: a domain with both x1 walls needs more "
                         "than 2 N_GHOSTS active cells in x1",
                         HERE);
          using klo_t = decltype(make(std::false_type {}, i_lo));
          using khi_t = decltype(make(std::true_type {}, i_hi));
          Kokkos::parallel_for(
            name,
            WallRange<D>(domain, rows_lo + rows_hi, active_transverse),
            BothWalls_kernel<klo_t, khi_t> { make(std::false_type {}, i_lo),
                                             make(std::true_type {}, i_hi),
                                             static_cast<cellidx_t>(rows_lo) });
        } else if (has_lo) {
          Kokkos::parallel_for(name,
                               WallRange<D>(domain, rows_lo, active_transverse),
                               make(std::false_type {}, i_lo));
        } else if (has_hi) {
          Kokkos::parallel_for(name,
                               WallRange<D>(domain, rows_hi, active_transverse),
                               make(std::true_type {}, i_hi));
        }
      }

    } // namespace wall

    /**
     * @brief Apply non-periodic field boundaries to the evolved `em` fields
     *        (perfect-conductor x1 walls): BC::E -> Ee (comps 0..2): E_tan = 0
     *        on the wall plane + mirror ghosts; BC::B -> Bf (comps 3..5): even
     *        mirror ghosts; the wall-plane B_n is left to Faraday (frozen by
     *        E_tan = 0) at -x1 and set from div B = 0 of the last active cell
     *        at +x1.
     * @param global_grid the global grid (for the configured boundary type)
     * @param tags        which fields to act on (BC::E, BC::B, or both)
     */
    template <Dimension D>
    void FieldBoundaries(Domain<SimEngine::HYBRID, metric::Minkowski<D>>& domain,
                         const Grid<D>& global_grid,
                         BCTags         tags) {
      auto& em = domain.fields.em;
      if (tags & BC::E) {
        wall::LaunchX1Walls<D>(
          "WallEdgeE",
          domain,
          global_grid,
          wall::staggered_rows_lo,
          wall::staggered_rows_hi,
          false,
          [&](auto plus, ncells_t i_edge) {
            return kernel::hybrid::WallEdgeE_kernel<D, decltype(plus)::value>(em,
                                                                              i_edge,
                                                                              0);
          });
      }
      if (tags & BC::B) {
        wall::LaunchX1Walls<D>(
          "WallFaceB",
          domain,
          global_grid,
          wall::staggered_rows_lo,
          wall::staggered_rows_hi,
          false,
          [&](auto plus, ncells_t i_edge) {
            return kernel::hybrid::WallFaceB_kernel<D, decltype(plus)::value, 6>(
              em,
              i_edge,
              3);
          });
      }
    }

    /**
     * @brief Conductor wall condition on the half-step edge-E (em0::345 =
     *        Ee'/Ee''). Call after EMF #1/#2, before the Faraday push that
     *        consumes it.
     */
    template <Dimension D>
    void WallEPrime(Domain<SimEngine::HYBRID, metric::Minkowski<D>>& domain,
                    const Grid<D>&                                   global_grid) {
      auto& em0 = domain.fields.em0;
      wall::LaunchX1Walls<D>(
        "WallEdgeE",
        domain,
        global_grid,
        wall::staggered_rows_lo,
        wall::staggered_rows_hi,
        false,
        [&](auto plus, ncells_t i_edge) {
          return kernel::hybrid::WallEdgeE_kernel<D, decltype(plus)::value>(em0,
                                                                            i_edge,
                                                                            3);
        });
    }

    /**
     * @brief Even-mirror the wall ghosts of the predictor face-B scratch
     *        (cur = Bf*, Bf**). Call after each Faraday #1/#2 (+ halo
     *        exchange), before the EMF that reads it: the Hall stencil reads
     *        the wall ghosts, which no halo exchange fills (left at zero they
     *        fake a wall current sheet J ~ B/dx every substage).
     */
    template <Dimension D>
    void WallScratchB(Domain<SimEngine::HYBRID, metric::Minkowski<D>>& domain,
                      const Grid<D>& global_grid) {
      auto& cur = domain.fields.cur;
      wall::LaunchX1Walls<D>(
        "WallFaceB",
        domain,
        global_grid,
        wall::staggered_rows_lo,
        wall::staggered_rows_hi,
        false,
        [&](auto plus, ncells_t i_edge) {
          return kernel::hybrid::WallFaceB_kernel<D, decltype(plus)::value, 3>(
            cur,
            i_edge,
            0);
        });
    }

    /**
     * @brief Conductor-sign mirror fill of the wall ghosts of the
     *        cell-centered gather fields (bckp: Ec' 0..2, Bc' 3..5). Call
     *        after each EMF #1/#2 (+ halo exchange), before the particle push
     *        that gathers from bckp.
     */
    template <Dimension D>
    void WallBckpFill(Domain<SimEngine::HYBRID, metric::Minkowski<D>>& domain,
                      const Grid<D>& global_grid) {
      auto& bckp = domain.fields.bckp;
      wall::LaunchX1Walls<D>(
        "WallBckp",
        domain,
        global_grid,
        N_GHOSTS,
        N_GHOSTS,
        false,
        [&](auto plus, ncells_t i_edge) {
          return kernel::hybrid::WallBckp_kernel<D, decltype(plus)::value>(bckp,
                                                                           i_edge);
        });
    }

    /**
     * @brief Mirror fill of the wall ghosts of a 6-component moments buffer
     *        (V_x odd, V_y/V_z/N even) from its active cells, over the full
     *        transverse extent: the fill half of MomentsWallBC for any buffer.
     */
    template <Dimension D>
    void MomentsWallFill(Domain<SimEngine::HYBRID, metric::Minkowski<D>>& domain,
                         const Grid<D>&                                   global_grid,
                         ndfield_t<D, 6>&                                 fld) {
      wall::LaunchX1Walls<D>(
        "WallMomentsFill",
        domain,
        global_grid,
        N_GHOSTS,
        N_GHOSTS,
        false,
        [&](auto plus, ncells_t i_edge) {
          return kernel::hybrid::WallMoments_kernel<D, decltype(plus)::value, false>(
            fld,
            i_edge);
        });
    }

    /**
     * @brief Image-plasma wall treatment of the deposited moments (aux).
     * @param fold also fold the ghost deposit tails back into the active cells
     *        (V_x sign-flipped) -- exactly once per deposit, after
     *        SynchronizeFields(AUX) and before CommunicateFields(AUX), so the
     *        transverse halo copies of the near-wall cells carry the folded
     *        values; fold and ghost fill run in one kernel. With fold = false
     *        only the (idempotent) ghost mirror fill runs, e.g. to re-fill
     *        after the halo exchange or a filter pass.
     * @param fold_halo fold over the transverse halo ghosts too: after
     *        SynchronizeFillFields they hold their owners' sums, so the fold
     *        there repeats the owners' fold
     */
    template <Dimension D>
    void MomentsWallBC(Domain<SimEngine::HYBRID, metric::Minkowski<D>>& domain,
                       const Grid<D>& global_grid,
                       bool           fold,
                       bool           fold_halo = false) {
      auto& aux = domain.fields.aux;
      if (fold) {
        // transverse extent of the fold: the active cells, or all with fold_halo
        ncells_t t_lo[2] { 0, 0 }, t_hi[2] { 1, 1 };
        const in axes[2] { in::x2, in::x3 };
        for (auto d { 1u }; d < static_cast<unsigned>(D); ++d) {
          t_lo[d - 1] = fold_halo ? 0 : domain.mesh.i_min(axes[d - 1]);
          t_hi[d - 1] = fold_halo ? domain.mesh.n_all(axes[d - 1])
                                  : domain.mesh.i_max(axes[d - 1]);
        }
        wall::LaunchX1Walls<D>(
          "WallMomentsFoldFill",
          domain,
          global_grid,
          N_GHOSTS,
          N_GHOSTS,
          false,
          [&](auto plus, ncells_t i_edge) {
            return kernel::hybrid::WallMomentsFoldFill_kernel<D, decltype(plus)::value>(
              aux,
              i_edge,
              t_lo,
              t_hi);
          });
      } else {
        MomentsWallFill<D>(domain, global_grid, aux);
      }
    }

  } // namespace hybrid
} // namespace ntt

#endif // ENGINES_HYBRID_FIELDS_BCS_H
