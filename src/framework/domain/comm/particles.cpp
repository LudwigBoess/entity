#include "enums.h"
#include "global.h"

#include "arch/directions.h"
#include "arch/kokkos_aliases.h"
#include "traits/metric.h"
#include "utils/log.h"

#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"
#include "framework/specialization_registry.h"

#include "framework/domain/comm/utils.hpp"

#if defined(MPI_ENABLED)
  #include "arch/mpi_tags.h"
#endif

#include <Kokkos_Core.hpp>

#include <string>
#include <utility>
#include <vector>

namespace ntt {

  template <SimEngine::type S, MetricClass M>
  void Metadomain<S, M>::CommunicateParticles(Domain<S, M>& domain) const {
#if defined(MPI_ENABLED)
    logger::Checkpoint("Communicating particles\n", HERE);
    // the directions requiring communication (the same for every species),
    // their partner ranks, and the cell-index shifts of the particles leaving
    // in them: + n of the target when leaving backwards, - n of the source
    // when leaving forwards
    PrtlCommPlan plan;
    for (const auto& direction : dir::Directions<D>::all) {
      // tag of the particles leaving in this direction
      const auto tag_send = mpi::PrtlSendTag<D>::dir2tag(direction);

      // get indices & ranks of send/recv meshblocks
      const auto [send_params,
                  recv_params] = GetSendRecvRanks(this, domain, direction);
      const auto [send_ind, send_rank] = send_params;
      const auto [recv_ind, recv_rank] = recv_params;

      // skip if no communication is necessary
      const auto is_sending   = (send_rank >= 0);
      const auto is_receiving = (recv_rank >= 0);
      if (not is_sending and not is_receiving) {
        continue;
      }
      PrtlCommPlan::Direction dr;
      dr.tag       = tag_send;
      dr.send_rank = send_rank;
      dr.recv_rank = recv_rank;
      dr.local     = is_sending and is_receiving and
                 (send_ind == domain.index()) and (recv_ind == domain.index());
      plan.dirs.push_back(dr);

      if (is_sending) {
        if constexpr (D == Dim::_1D || D == Dim::_2D || D == Dim::_3D) {
          if (direction[0] == -1) {
            plan.shift1[tag_send] = subdomain(send_ind).mesh.n_active(in::x1);
          } else if (direction[0] == 1) {
            plan.shift1[tag_send] = -domain.mesh.n_active(in::x1);
          }
        }
        if constexpr (D == Dim::_2D || D == Dim::_3D) {
          if (direction[1] == -1) {
            plan.shift2[tag_send] = subdomain(send_ind).mesh.n_active(in::x2);
          } else if (direction[1] == 1) {
            plan.shift2[tag_send] = -domain.mesh.n_active(in::x2);
          }
        }
        if constexpr (D == Dim::_3D) {
          if (direction[2] == -1) {
            plan.shift3[tag_send] = subdomain(send_ind).mesh.n_active(in::x3);
          } else if (direction[2] == 1) {
            plan.shift3[tag_send] = -domain.mesh.n_active(in::x3);
          }
        }
      }
    }
    for (auto& species : domain.species) {
      species.Communicate(plan);
    }
#else
    (void)domain;
#endif
  }

  // NOLINTBEGIN(bugprone-macro-parentheses)
#define METADOMAIN_COMM(S, M, D)                                               \
  template void Metadomain<S, M<D>>::CommunicateParticles(Domain<S, M<D>>&) const;

  NTT_FOREACH_SPECIALIZATION(METADOMAIN_COMM)
#undef METADOMAIN_COMM
  // NOLINTEND(bugprone-macro-parentheses)

} // namespace ntt
