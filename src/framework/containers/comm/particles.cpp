#include "framework/containers/particles.h"

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/mpi_aliases.h"
#include "arch/mpi_tags.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"

#include "framework/specialization_registry.h"
#include "kernels/comm.hpp"

#include <mpi.h>

#include <climits>
#include <string>
#include <utility>
#include <vector>

namespace ntt {

  namespace prtls {
    // message tags of the particle exchange: base + 32 * buffer type +
    // direction index (tag - 2 < 26); the count messages use type 5
    constexpr int TAG_BASE = 2048;

    enum BufferType : int {
      Int = 0,
      Real,
      Prtldx,
      PldR,
      PldI,
      Count
    };

    inline auto message_tag(int type, short prtl_tag) -> int {
      return TAG_BASE + 32 * type + (static_cast<int>(prtl_tag) - 2);
    }

    // grow-only (re)allocation of a persistent buffer to hold `n` elements,
    // with 1/8 headroom
    template <typename T>
    void reserve(array_t<T*>& v, const char* label, npart_t n) {
      if (static_cast<npart_t>(v.extent(0)) < n) {
        v = array_t<T*> {};
        v = array_t<T*> { Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                             std::string { label }),
                          n + n / 8u };
      }
    }

    // buffer handed to MPI: the device buffer itself, or a host copy without
    // GPU-aware MPI
#if defined(DEVICE_ENABLED) && !defined(GPU_AWARE_MPI)
    template <typename T>
    using mpi_buf_t = Kokkos::View<T*, Kokkos::HostSpace>;
#else
    template <typename T>
    using mpi_buf_t = array_t<T*>;
#endif

    // the MPI buffer of the first `n` elements of `buf` (`to_host`: with
    // their current values)
    template <typename T>
    auto mpi_view(const array_t<T*>& buf, npart_t n, bool to_host)
      -> mpi_buf_t<T> {
#if defined(DEVICE_ENABLED) && !defined(GPU_AWARE_MPI)
      mpi_buf_t<T> h { Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                          std::string { "prtl_comm_host" }),
                       n };
      if (to_host and n > 0u) {
        Kokkos::deep_copy(
          h,
          Kokkos::subview(buf, std::make_pair(static_cast<npart_t>(0), n)));
      }
      return h;
#else
      (void)n;
      (void)to_host;
      return buf;
#endif
    }

    // copy the received values back to the device buffer
    template <typename T>
    void from_mpi_view(const array_t<T*>& buf, const mpi_buf_t<T>& h, npart_t n) {
#if defined(DEVICE_ENABLED) && !defined(GPU_AWARE_MPI)
      if (n > 0u) {
        Kokkos::deep_copy(
          Kokkos::subview(buf, std::make_pair(static_cast<npart_t>(0), n)),
          h);
      }
#else
      (void)buf;
      (void)h;
      (void)n;
#endif
    }

    // nonblocking send / receive of `count` elements at `buf + off`
    template <typename T>
    void isend(const mpi_buf_t<T>&       buf,
               npart_t                   off,
               npart_t                   count,
               int                       rank,
               int                       tag,
               std::vector<MPI_Request>& reqs) {
      if (count == 0u) {
        return;
      }
      raise::ErrorIf(count > static_cast<npart_t>(INT_MAX),
                     "particle message too large",
                     HERE);
      reqs.push_back(MPI_REQUEST_NULL);
      MPI_Isend(buf.data() + off,
                static_cast<int>(count),
                mpi::get_type<T>(),
                rank,
                tag,
                MPI_COMM_WORLD,
                &reqs.back());
    }

    template <typename T>
    void irecv(const mpi_buf_t<T>&       buf,
               npart_t                   off,
               npart_t                   count,
               int                       rank,
               int                       tag,
               std::vector<MPI_Request>& reqs) {
      if (count == 0u) {
        return;
      }
      raise::ErrorIf(count > static_cast<npart_t>(INT_MAX),
                     "particle message too large",
                     HERE);
      reqs.push_back(MPI_REQUEST_NULL);
      MPI_Irecv(buf.data() + off,
                static_cast<int>(count),
                mpi::get_type<T>(),
                rank,
                tag,
                MPI_COMM_WORLD,
                &reqs.back());
    }
  } // namespace prtls

  template <Dimension D, Coord::type C>
  void Particles<D, C>::Communicate(const PrtlCommPlan& plan) {
    logger::Checkpoint(fmt::format("Communicating species #%d\n", index()), HERE);

    using kernel::comm::prtl_tag_offsets_t;
    using kernel::comm::prtl_tag_table_t;

    const auto nt = static_cast<short>(ntags());
    raise::ErrorIf(static_cast<std::size_t>(nt) > mpi::MaxPrtlTags,
                   "too many particle tags",
                   HERE);

    // tags wrapped in place (periodic self-wrap) and tags exchanged with
    // other ranks
    prtl_tag_table_t local {}, packed {};
    bool             any_local = false, any_remote = false;
    for (const auto& dr : plan.dirs) {
      if (dr.local) {
        local[dr.tag] = 1;
        any_local     = true;
      } else {
        packed[dr.tag] = 1;
        any_remote     = true;
      }
    }

    // 1. wrap the self-wrapping particles in place and, with remote
    //    directions, count every other non-alive tag
    if (any_remote) {
      if (static_cast<short>(m_comm_counts.extent(0)) < nt) {
        m_comm_counts = array_t<npart_t*> { "prtl_comm_counts",
                                            static_cast<std::size_t>(nt) };
        m_comm_cursor = array_t<npart_t*> { "prtl_comm_cursor",
                                            static_cast<std::size_t>(nt) };
      } else {
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace {},
                          m_comm_counts,
                          static_cast<npart_t>(0));
      }
    }
    if ((any_local or any_remote) and npart() > 0u) {
      // clang-format off
      Kokkos::parallel_for(
        "WrapAndCountPrtlTags",
        rangeActiveParticles(),
        kernel::comm::WrapAndCountPrtlTags_kernel<D>(
          i1, i1_prev, i2, i2_prev, i3, i3_prev, tag, m_comm_counts,
          local, plan.shift1, plan.shift2, plan.shift3, nt, any_remote));
      // clang-format on
    }
    if (not any_remote) {
      set_unsorted();
      return;
    }

    // per-tag counts on the host (the copy waits for the count kernel)
    std::vector<npart_t> npptag(static_cast<std::size_t>(nt), 0u);
    {
      auto counts_h = Kokkos::create_mirror_view(m_comm_counts);
      Kokkos::deep_copy(counts_h, m_comm_counts);
      for (short t = 0; t < nt; ++t) {
        npptag[t] = (t == ParticleTag::alive) ? 0u : counts_h(t);
      }
    }
    const auto npart_dead = npptag[ParticleTag::dead];
    npart_t    nholes     = 0u;
    for (short t = 0; t < nt; ++t) {
      nholes += npptag[t];
    }

    // 2. exchange the counts with all partners at once
    const auto               ndirs = plan.dirs.size();
    std::vector<npart_t>     nrecv(ndirs, 0u);
    std::vector<MPI_Request> reqs;
    reqs.reserve(2u * ndirs * static_cast<std::size_t>(prtls::Count + 1));
    for (auto k { 0u }; k < ndirs; ++k) {
      const auto& dr = plan.dirs[k];
      if (dr.local) {
        continue;
      }
      if (dr.recv_rank >= 0) {
        reqs.push_back(MPI_REQUEST_NULL);
        MPI_Irecv(&nrecv[k],
                  1,
                  mpi::get_type<npart_t>(),
                  dr.recv_rank,
                  prtls::message_tag(prtls::Count, dr.tag),
                  MPI_COMM_WORLD,
                  &reqs.back());
      }
      if (dr.send_rank >= 0) {
        reqs.push_back(MPI_REQUEST_NULL);
        MPI_Isend(&npptag[dr.tag],
                  1,
                  mpi::get_type<npart_t>(),
                  dr.send_rank,
                  prtls::message_tag(prtls::Count, dr.tag),
                  MPI_COMM_WORLD,
                  &reqs.back());
      }
    }
    MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    reqs.clear();

    // receive offsets in direction order
    std::vector<npart_t> recv_off(ndirs, 0u);
    npart_t              npart_recv = 0u;
    for (auto k { 0u }; k < ndirs; ++k) {
      recv_off[k]  = npart_recv;
      npart_recv  += nrecv[k];
    }
    raise::ErrorIf((npart() + npart_recv) >= maxnpart(),
                   "Too many particles to receive (cannot fit into maxptl)",
                   HERE);

    // 3. slots of the non-alive particles: the dead first, then the leaving
    //    ones in tag order (`start[t]` is the first slot of tag t)
    prtl_tag_offsets_t start {};
    {
      npart_t s = npart_dead;
      for (short t = 2; t < nt; ++t) {
        start[t]  = s;
        s        += npptag[t];
      }
    }
    prtls::reserve(m_comm_holes, "prtl_comm_holes", nholes);
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace {},
                      m_comm_cursor,
                      static_cast<npart_t>(0));
    // clang-format off
    Kokkos::parallel_for(
      "PrepareOutgoingPrtls",
      rangeActiveParticles(),
      kernel::comm::PrepareOutgoingPrtls_kernel<D>(
          i1, i1_prev, i2, i2_prev, i3, i3_prev, tag,
          m_comm_holes, m_comm_cursor,
          start, plan.shift1, plan.shift2, plan.shift3, nholes)
    );
    // clang-format on

    // number of values of each type per particle
    const unsigned short NREALS = 4 + static_cast<unsigned short>(
                                        D == Dim::_2D and C != Coord::Cartesian);
    const unsigned short NINTS   = 2 * static_cast<unsigned short>(D);
    const unsigned short NPRTLDX = 2 * static_cast<unsigned short>(D);
    const unsigned short NPLDS_R = npld_r();
    const unsigned short NPLDS_I = npld_i();

    // 4. pack every leaving particle at its position in the slot list after
    //    the dead: the particles of one direction form one segment
    const npart_t nsend = nholes - npart_dead;
    prtls::reserve(m_comm_send_int, "prtl_send_int", nsend * NINTS);
    prtls::reserve(m_comm_send_real, "prtl_send_real", nsend * NREALS);
    prtls::reserve(m_comm_send_prtldx, "prtl_send_prtldx", nsend * NPRTLDX);
    prtls::reserve(m_comm_recv_int, "prtl_recv_int", npart_recv * NINTS);
    prtls::reserve(m_comm_recv_real, "prtl_recv_real", npart_recv * NREALS);
    prtls::reserve(m_comm_recv_prtldx, "prtl_recv_prtldx", npart_recv * NPRTLDX);
    if (NPLDS_R > 0) {
      prtls::reserve(m_comm_send_pld_r, "prtl_send_pld_r", nsend * NPLDS_R);
      prtls::reserve(m_comm_recv_pld_r, "prtl_recv_pld_r", npart_recv * NPLDS_R);
    }
    if (NPLDS_I > 0) {
      prtls::reserve(m_comm_send_pld_i, "prtl_send_pld_i", nsend * NPLDS_I);
      prtls::reserve(m_comm_recv_pld_i, "prtl_recv_pld_i", npart_recv * NPLDS_I);
    }
    if (nsend > 0u) {
      // clang-format off
      Kokkos::parallel_for(
        "PopulatePrtlSendBuffer",
        nsend,
        kernel::comm::PopulatePrtlSendBuffer_kernel<D, C>(
          m_comm_send_int, m_comm_send_real, m_comm_send_prtldx,
          m_comm_send_pld_r, m_comm_send_pld_i,
          NINTS, NREALS, NPRTLDX, NPLDS_R, NPLDS_I, npart_dead,
          i1, i1_prev, dx1, dx1_prev,
          i2, i2_prev, dx2, dx2_prev,
          i3, i3_prev, dx3, dx3_prev,
          ux1, ux2, ux3,
          weight, phi, pld_r, pld_i, tag,
          m_comm_holes, packed)
      );
      // clang-format on
    }

    // 5. all data messages in flight at once
#if defined(DEVICE_ENABLED)
    // Intel GPUs: every kernel writing a send buffer must complete before MPI
    // reads it
    Kokkos::fence("CommunicateParticles: before MPI");
#endif
    const auto s_int  = prtls::mpi_view(m_comm_send_int, nsend * NINTS, true);
    const auto s_real = prtls::mpi_view(m_comm_send_real, nsend * NREALS, true);
    const auto s_dx = prtls::mpi_view(m_comm_send_prtldx, nsend * NPRTLDX, true);
    const auto r_int = prtls::mpi_view(m_comm_recv_int, npart_recv * NINTS, false);
    const auto r_real = prtls::mpi_view(m_comm_recv_real, npart_recv * NREALS, false);
    const auto r_dx = prtls::mpi_view(m_comm_recv_prtldx, npart_recv * NPRTLDX, false);
    prtls::mpi_buf_t<real_t>  s_pr, r_pr;
    prtls::mpi_buf_t<npart_t> s_pi, r_pi;
    if (NPLDS_R > 0) {
      s_pr = prtls::mpi_view(m_comm_send_pld_r, nsend * NPLDS_R, true);
      r_pr = prtls::mpi_view(m_comm_recv_pld_r, npart_recv * NPLDS_R, false);
    }
    if (NPLDS_I > 0) {
      s_pi = prtls::mpi_view(m_comm_send_pld_i, nsend * NPLDS_I, true);
      r_pi = prtls::mpi_view(m_comm_recv_pld_i, npart_recv * NPLDS_I, false);
    }
    for (auto k { 0u }; k < ndirs; ++k) {
      const auto& dr = plan.dirs[k];
      if (dr.local or dr.recv_rank < 0 or nrecv[k] == 0u) {
        continue;
      }
      const auto o = recv_off[k], n = nrecv[k];
      const auto r = dr.recv_rank;
      using namespace prtls;
      irecv(r_int, o * NINTS, n * NINTS, r, message_tag(Int, dr.tag), reqs);
      irecv(r_real, o * NREALS, n * NREALS, r, message_tag(Real, dr.tag), reqs);
      irecv(r_dx, o * NPRTLDX, n * NPRTLDX, r, message_tag(Prtldx, dr.tag), reqs);
      if (NPLDS_R > 0) {
        irecv(r_pr, o * NPLDS_R, n * NPLDS_R, r, message_tag(PldR, dr.tag), reqs);
      }
      if (NPLDS_I > 0) {
        irecv(r_pi, o * NPLDS_I, n * NPLDS_I, r, message_tag(PldI, dr.tag), reqs);
      }
    }
    for (const auto& dr : plan.dirs) {
      if (dr.local or dr.send_rank < 0 or npptag[dr.tag] == 0u) {
        continue;
      }
      const auto o = start[dr.tag] - npart_dead, n = npptag[dr.tag];
      const auto r = dr.send_rank;
      using namespace prtls;
      isend(s_int, o * NINTS, n * NINTS, r, message_tag(Int, dr.tag), reqs);
      isend(s_real, o * NREALS, n * NREALS, r, message_tag(Real, dr.tag), reqs);
      isend(s_dx, o * NPRTLDX, n * NPRTLDX, r, message_tag(Prtldx, dr.tag), reqs);
      if (NPLDS_R > 0) {
        isend(s_pr, o * NPLDS_R, n * NPLDS_R, r, message_tag(PldR, dr.tag), reqs);
      }
      if (NPLDS_I > 0) {
        isend(s_pi, o * NPLDS_I, n * NPLDS_I, r, message_tag(PldI, dr.tag), reqs);
      }
    }
    MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    prtls::from_mpi_view(m_comm_recv_int, r_int, npart_recv * NINTS);
    prtls::from_mpi_view(m_comm_recv_real, r_real, npart_recv * NREALS);
    prtls::from_mpi_view(m_comm_recv_prtldx, r_dx, npart_recv * NPRTLDX);
    if (NPLDS_R > 0) {
      prtls::from_mpi_view(m_comm_recv_pld_r, r_pr, npart_recv * NPLDS_R);
    }
    if (NPLDS_I > 0) {
      prtls::from_mpi_view(m_comm_recv_pld_i, r_pi, npart_recv * NPLDS_I);
    }

    // 6. unpack: the holes first, then append
    if (npart_recv > 0u) {
      // clang-format off
      Kokkos::parallel_for(
        "PopulateFromRecvBuffer",
        npart_recv,
        kernel::comm::ExtractReceivedPrtls_kernel<D, C>(
              m_comm_recv_int, m_comm_recv_real, m_comm_recv_prtldx,
              m_comm_recv_pld_r, m_comm_recv_pld_i,
              NINTS, NREALS, NPRTLDX, NPLDS_R, NPLDS_I,
              npart(), nholes,
              i1, i1_prev, dx1, dx1_prev,
              i2, i2_prev, dx2, dx2_prev,
              i3, i3_prev, dx3, dx3_prev,
              ux1, ux2, ux3,
              weight, phi, pld_r, pld_i, tag,
              m_comm_holes)
      );
      // clang-format on
    }

    if (npart_recv > nholes) {
      set_npart(npart() + npart_recv - nholes);
    }
    set_unsorted();
  }

#define PARTICLES_COMM(D, C)                                                   \
  template void Particles<D, C>::Communicate(const PrtlCommPlan&);

  NTT_FOREACH_COORDINATE(PARTICLES_COMM)
#undef PARTICLES_COMM

} // namespace ntt
