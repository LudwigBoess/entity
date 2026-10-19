/**
 * @file engines/hybrid/moments_remap.h
 * @brief Ghost remap of the deposited hybrid ion moments
 * @implements
 *   - ntt::hybrid::RemapMoments<> -> void
 * @namespaces:
 *   - ntt::hybrid::
 * @macros:
 *   - MPI_ENABLED
 */

#ifndef ENGINES_HYBRID_MOMENTS_REMAP_H
#define ENGINES_HYBRID_MOMENTS_REMAP_H

#include "enums.h"
#include "global.h"

#include "utils/timer.h"

#include "metrics/minkowski.h"

#include "engines/hybrid/fields_bcs.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

namespace ntt {
  namespace hybrid {

    /**
     * @brief Remap the moments (aux comps 0..3 = V, N) after a deposit: add
     *        the halo deposit tails into the owned cells, fold the wall ghost
     *        tails back (image plasma) and fill every ghost cell for the EMF
     *        and filter stencils.
     *
     * With MPI one exchange sums and fills the halo (SynchronizeFillFields:
     * every halo ghost holds the bitwise copy of its owner's sum), then the
     * wall fold runs over the halo ghosts too, repeating the owners' fold.
     * Without MPI: sum, fold, fill as separate steps. Both give the same
     * values.
     */
    template <Dimension D>
    void RemapMoments(Metadomain<SimEngine::HYBRID, metric::Minkowski<D>>& metadomain,
                      Domain<SimEngine::HYBRID, metric::Minkowski<D>>& domain,
                      timer::Timers&                                   timers) {
#if defined(MPI_ENABLED)
      timers.start("Communications");
      metadomain.SynchronizeFillFields(domain, ::Comm::AUX_0123);
      timers.stop("Communications");
      timers.start("FieldBoundaries");
      MomentsWallBC(domain, metadomain.mesh(), /* fold */ true, /* fold_halo */ true);
      timers.stop("FieldBoundaries");
#else
      timers.start("Communications");
      metadomain.SynchronizeFields(domain, ::Comm::AUX_0123);
      timers.stop("Communications");
      // fold before the ghost fill, so the halo copies carry the folded values
      timers.start("FieldBoundaries");
      MomentsWallBC(domain, metadomain.mesh(), /* fold */ true);
      timers.stop("FieldBoundaries");
      timers.start("Communications");
      metadomain.CommunicateFields(domain, ::Comm::AUX_0123);
      timers.stop("Communications");
      // mirror-fill the wall ghosts again, now that the transverse halo is filled
      timers.start("FieldBoundaries");
      MomentsWallBC(domain, metadomain.mesh(), /* fold */ false);
      timers.stop("FieldBoundaries");
#endif
    }

  } // namespace hybrid
} // namespace ntt

#endif // ENGINES_HYBRID_MOMENTS_REMAP_H
