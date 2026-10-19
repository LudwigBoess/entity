/**
 * @file kernels/comm.hpp
 * @brief Kernels used during communications
 * @implements
 *   - kernel::comm::WrapAndCountPrtlTags_kernel<>
 *   - kernel::comm::PrepareOutgoingPrtls_kernel<>
 *   - kernel::comm::PopulatePrtlSendBuffer_kernel<>
 *   - kernel::comm::ExtractReceivedPrtls_kernel<>
 * @namespaces:
 *   - kernel::comm::
 */

#ifndef KERNELS_COMM_HPP
#define KERNELS_COMM_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/mpi_tags.h"

#include <Kokkos_Core.hpp>

namespace kernel::comm {
  using namespace ntt;

  // per-tag tables passed by value (indexed by the particle tag)
  using prtl_tag_table_t   = Kokkos::Array<int, mpi::MaxPrtlTags>;
  using prtl_tag_offsets_t = Kokkos::Array<npart_t, mpi::MaxPrtlTags>;

  /**
   * @brief Wraps the particles leaving through a periodic self-wrap (`local`
   *        tags) in place: cell indices shifted by the domain size, tag back
   *        to alive. With `count`, counts the particles of every other
   *        non-alive tag into `counts`.
   */
  template <Dimension D>
  class WrapAndCountPrtlTags_kernel {
    array_t<int*>          i1, i1_prev, i2, i2_prev, i3, i3_prev;
    array_t<short*>        tag;
    array_t<npart_t*>      counts;
    const prtl_tag_table_t local, shift1, shift2, shift3;
    const short            ntags;
    const bool             count;

  public:
    WrapAndCountPrtlTags_kernel(array_t<int*>&           i1,
                                array_t<int*>&           i1_prev,
                                array_t<int*>&           i2,
                                array_t<int*>&           i2_prev,
                                array_t<int*>&           i3,
                                array_t<int*>&           i3_prev,
                                array_t<short*>&         tag,
                                const array_t<npart_t*>& counts,
                                const prtl_tag_table_t&  local,
                                const prtl_tag_table_t&  shift1,
                                const prtl_tag_table_t&  shift2,
                                const prtl_tag_table_t&  shift3,
                                short                    ntags,
                                bool                     count)
      : i1 { i1 }
      , i1_prev { i1_prev }
      , i2 { i2 }
      , i2_prev { i2_prev }
      , i3 { i3 }
      , i3_prev { i3_prev }
      , tag { tag }
      , counts { counts }
      , local { local }
      , shift1 { shift1 }
      , shift2 { shift2 }
      , shift3 { shift3 }
      , ntags { ntags }
      , count { count } {}

    Inline void operator()(prtlidx_t p) const {
      const short t = tag(p);
      if (t < 0 or t >= ntags) {
        raise::KernelError(HERE, "Invalid tag value");
      }
      if (t == ParticleTag::alive) {
        return;
      }
      if (local[t] != 0) {
        if constexpr (D == Dim::_1D or D == Dim::_2D or D == Dim::_3D) {
          i1(p)      += shift1[t];
          i1_prev(p) += shift1[t];
        }
        if constexpr (D == Dim::_2D or D == Dim::_3D) {
          i2(p)      += shift2[t];
          i2_prev(p) += shift2[t];
        }
        if constexpr (D == Dim::_3D) {
          i3(p)      += shift3[t];
          i3_prev(p) += shift3[t];
        }
        tag(p) = ParticleTag::alive;
      } else if (count) {
        Kokkos::atomic_add(&counts(t), static_cast<npart_t>(1));
      }
    }
  };

  /**
   * @brief Lists the slots of the non-alive particles in `holes`: the dead
   *        ones first, then the leaving ones grouped by tag in tag order
   *        (`start[t]` is the first slot of tag t), and shifts the cell
   *        indices of every leaving particle into the frame of the neighbor
   *        it leaves to.
   */
  template <Dimension D>
  class PrepareOutgoingPrtls_kernel {
    array_t<int*>            i1, i1_prev, i2, i2_prev, i3, i3_prev;
    const array_t<short*>    tag;
    array_t<npart_t*>        holes;
    array_t<npart_t*>        cursor;
    const prtl_tag_offsets_t start;
    const prtl_tag_table_t   shift1, shift2, shift3;
    const npart_t            nholes;

  public:
    PrepareOutgoingPrtls_kernel(array_t<int*>&            i1,
                                array_t<int*>&            i1_prev,
                                array_t<int*>&            i2,
                                array_t<int*>&            i2_prev,
                                array_t<int*>&            i3,
                                array_t<int*>&            i3_prev,
                                const array_t<short*>&    tag,
                                const array_t<npart_t*>&  holes,
                                const array_t<npart_t*>&  cursor,
                                const prtl_tag_offsets_t& start,
                                const prtl_tag_table_t&   shift1,
                                const prtl_tag_table_t&   shift2,
                                const prtl_tag_table_t&   shift3,
                                npart_t                   nholes)
      : i1 { i1 }
      , i1_prev { i1_prev }
      , i2 { i2 }
      , i2_prev { i2_prev }
      , i3 { i3 }
      , i3_prev { i3_prev }
      , tag { tag }
      , holes { holes }
      , cursor { cursor }
      , start { start }
      , shift1 { shift1 }
      , shift2 { shift2 }
      , shift3 { shift3 }
      , nholes { nholes } {}

    Inline void operator()(prtlidx_t p) const {
      const short t = tag(p);
      if (t == ParticleTag::alive) {
        return;
      }
      const auto k = start[t] + Kokkos::atomic_fetch_add(&cursor(t),
                                                         static_cast<npart_t>(1));
      if (k >= nholes) {
        raise::KernelError(HERE, "Outgoing indices idx exceeds the array size");
      }
      holes(k) = p;
      if (t != ParticleTag::dead) {
        if constexpr (D == Dim::_1D or D == Dim::_2D or D == Dim::_3D) {
          i1(p)      += shift1[t];
          i1_prev(p) += shift1[t];
        }
        if constexpr (D == Dim::_2D or D == Dim::_3D) {
          i2(p)      += shift2[t];
          i2_prev(p) += shift2[t];
        }
        if constexpr (D == Dim::_3D) {
          i3(p)      += shift3[t];
          i3_prev(p) += shift3[t];
        }
      }
    }
  };

  /**
   * @brief Packs the leaving particles listed in `holes` from `idx_offset`
   *        on into the send buffers, at the position they have in that list
   *        (so the particles of one direction form one contiguous segment),
   *        and tags them dead. Particles whose tag belongs to no exchanged
   *        direction (`packed[tag] == 0`) are left untouched.
   */
  template <Dimension D, Coord::type C>
  class PopulatePrtlSendBuffer_kernel {
    array_t<int*>      send_buff_int;
    array_t<real_t*>   send_buff_real;
    array_t<prtldx_t*> send_buff_prtldx;
    array_t<real_t*>   send_buff_pld_r;
    array_t<npart_t*>  send_buff_pld_i;

    const unsigned short NINTS, NREALS, NPRTLDX, NPLDS_R, NPLDS_I;
    const npart_t        idx_offset;

    const array_t<int*>      i1, i1_prev, i2, i2_prev, i3, i3_prev;
    const array_t<prtldx_t*> dx1, dx1_prev, dx2, dx2_prev, dx3, dx3_prev;
    const array_t<real_t*>   ux1, ux2, ux3, weight, phi;
    const array_t<real_t**>  pld_r;
    const array_t<npart_t**> pld_i;
    array_t<short*>          tag;
    const array_t<npart_t*>  outgoing_indices;
    const prtl_tag_table_t   packed;

  public:
    PopulatePrtlSendBuffer_kernel(array_t<int*>&            send_buff_int,
                                  array_t<real_t*>&         send_buff_real,
                                  array_t<prtldx_t*>&       send_buff_prtldx,
                                  array_t<real_t*>&         send_buff_pld_r,
                                  array_t<npart_t*>&        send_buff_pld_i,
                                  unsigned short            NINTS,
                                  unsigned short            NREALS,
                                  unsigned short            NPRTLDX,
                                  unsigned short            NPLDS_R,
                                  unsigned short            NPLDS_I,
                                  npart_t                   idx_offset,
                                  const array_t<int*>&      i1,
                                  const array_t<int*>&      i1_prev,
                                  const array_t<prtldx_t*>& dx1,
                                  const array_t<prtldx_t*>& dx1_prev,
                                  const array_t<int*>&      i2,
                                  const array_t<int*>&      i2_prev,
                                  const array_t<prtldx_t*>& dx2,
                                  const array_t<prtldx_t*>& dx2_prev,
                                  const array_t<int*>&      i3,
                                  const array_t<int*>&      i3_prev,
                                  const array_t<prtldx_t*>& dx3,
                                  const array_t<prtldx_t*>& dx3_prev,
                                  const array_t<real_t*>&   ux1,
                                  const array_t<real_t*>&   ux2,
                                  const array_t<real_t*>&   ux3,
                                  const array_t<real_t*>&   weight,
                                  const array_t<real_t*>&   phi,
                                  const array_t<real_t**>&  pld_r,
                                  const array_t<npart_t**>& pld_i,
                                  array_t<short*>&          tag,
                                  const array_t<npart_t*>&  outgoing_indices,
                                  const prtl_tag_table_t&   packed)
      : send_buff_int { send_buff_int }
      , send_buff_real { send_buff_real }
      , send_buff_prtldx { send_buff_prtldx }
      , send_buff_pld_r { send_buff_pld_r }
      , send_buff_pld_i { send_buff_pld_i }
      , NINTS { NINTS }
      , NREALS { NREALS }
      , NPRTLDX { NPRTLDX }
      , NPLDS_R { NPLDS_R }
      , NPLDS_I { NPLDS_I }
      , idx_offset { idx_offset }
      , i1 { i1 }
      , i1_prev { i1_prev }
      , i2 { i2 }
      , i2_prev { i2_prev }
      , i3 { i3 }
      , i3_prev { i3_prev }
      , dx1 { dx1 }
      , dx1_prev { dx1_prev }
      , dx2 { dx2 }
      , dx2_prev { dx2_prev }
      , dx3 { dx3 }
      , dx3_prev { dx3_prev }
      , ux1 { ux1 }
      , ux2 { ux2 }
      , ux3 { ux3 }
      , weight { weight }
      , phi { phi }
      , pld_r { pld_r }
      , pld_i { pld_i }
      , tag { tag }
      , outgoing_indices { outgoing_indices }
      , packed { packed } {}

    Inline void operator()(prtlidx_t p) const {
      const auto idx = outgoing_indices(idx_offset + p);
      if (packed[tag(idx)] == 0) {
        return;
      }
      if constexpr (D == Dim::_1D or D == Dim::_2D or D == Dim::_3D) {
        send_buff_int(NINTS * p + 0)      = i1(idx);
        send_buff_int(NINTS * p + 1)      = i1_prev(idx);
        send_buff_prtldx(NPRTLDX * p + 0) = dx1(idx);
        send_buff_prtldx(NPRTLDX * p + 1) = dx1_prev(idx);
      }
      if constexpr (D == Dim::_2D or D == Dim::_3D) {
        send_buff_int(NINTS * p + 2)      = i2(idx);
        send_buff_int(NINTS * p + 3)      = i2_prev(idx);
        send_buff_prtldx(NPRTLDX * p + 2) = dx2(idx);
        send_buff_prtldx(NPRTLDX * p + 3) = dx2_prev(idx);
      }
      if constexpr (D == Dim::_3D) {
        send_buff_int(NINTS * p + 4)      = i3(idx);
        send_buff_int(NINTS * p + 5)      = i3_prev(idx);
        send_buff_prtldx(NPRTLDX * p + 4) = dx3(idx);
        send_buff_prtldx(NPRTLDX * p + 5) = dx3_prev(idx);
      }
      send_buff_real(NREALS * p + 0) = ux1(idx);
      send_buff_real(NREALS * p + 1) = ux2(idx);
      send_buff_real(NREALS * p + 2) = ux3(idx);
      send_buff_real(NREALS * p + 3) = weight(idx);
      if constexpr (D == Dim::_2D and C != Coord::Cartesian) {
        send_buff_real(NREALS * p + 4) = phi(idx);
      }
      if (NPLDS_R > 0) {
        for (auto l { 0u }; l < NPLDS_R; ++l) {
          send_buff_pld_r(NPLDS_R * p + l) = pld_r(idx, l);
        }
      }
      if (NPLDS_I > 0) {
        for (auto l { 0u }; l < NPLDS_I; ++l) {
          send_buff_pld_i(NPLDS_I * p + l) = pld_i(idx, l);
        }
      }
      tag(idx) = ParticleTag::dead;
    }
  };

  /**
   * @brief Unpacks the received particles: the first `npart_holes` go to the
   *        slots listed in `outgoing_indices`, the rest are appended after
   *        `npart`.
   */
  template <Dimension D, Coord::type C>
  class ExtractReceivedPrtls_kernel {
    const array_t<int*>      recv_buff_int;
    const array_t<real_t*>   recv_buff_real;
    const array_t<prtldx_t*> recv_buff_prtldx;
    const array_t<real_t*>   recv_buff_pld_r;
    const array_t<npart_t*>  recv_buff_pld_i;

    const unsigned short NINTS, NREALS, NPRTLDX, NPLDS_R, NPLDS_I;
    const npart_t        npart, npart_holes;

    array_t<int*>           i1, i1_prev, i2, i2_prev, i3, i3_prev;
    array_t<prtldx_t*>      dx1, dx1_prev, dx2, dx2_prev, dx3, dx3_prev;
    array_t<real_t*>        ux1, ux2, ux3, weight, phi;
    array_t<real_t**>       pld_r;
    array_t<npart_t**>      pld_i;
    array_t<short*>         tag;
    const array_t<npart_t*> outgoing_indices;

  public:
    ExtractReceivedPrtls_kernel(const array_t<int*>&      recv_buff_int,
                                const array_t<real_t*>&   recv_buff_real,
                                const array_t<prtldx_t*>& recv_buff_prtldx,
                                const array_t<real_t*>&   recv_buff_pld_r,
                                const array_t<npart_t*>&  recv_buff_pld_i,
                                unsigned short            NINTS,
                                unsigned short            NREALS,
                                unsigned short            NPRTLDX,
                                unsigned short            NPLDS_R,
                                unsigned short            NPLDS_I,
                                npart_t                   npart,
                                npart_t                   npart_holes,
                                array_t<int*>&            i1,
                                array_t<int*>&            i1_prev,
                                array_t<prtldx_t*>&       dx1,
                                array_t<prtldx_t*>&       dx1_prev,
                                array_t<int*>&            i2,
                                array_t<int*>&            i2_prev,
                                array_t<prtldx_t*>&       dx2,
                                array_t<prtldx_t*>&       dx2_prev,
                                array_t<int*>&            i3,
                                array_t<int*>&            i3_prev,
                                array_t<prtldx_t*>&       dx3,
                                array_t<prtldx_t*>&       dx3_prev,
                                array_t<real_t*>&         ux1,
                                array_t<real_t*>&         ux2,
                                array_t<real_t*>&         ux3,
                                array_t<real_t*>&         weight,
                                array_t<real_t*>&         phi,
                                array_t<real_t**>&        pld_r,
                                array_t<npart_t**>&       pld_i,
                                array_t<short*>&          tag,
                                const array_t<npart_t*>&  outgoing_indices)
      : recv_buff_int { recv_buff_int }
      , recv_buff_real { recv_buff_real }
      , recv_buff_prtldx { recv_buff_prtldx }
      , recv_buff_pld_r { recv_buff_pld_r }
      , recv_buff_pld_i { recv_buff_pld_i }
      , NINTS { NINTS }
      , NREALS { NREALS }
      , NPRTLDX { NPRTLDX }
      , NPLDS_R { NPLDS_R }
      , NPLDS_I { NPLDS_I }
      , npart { npart }
      , npart_holes { npart_holes }
      , i1 { i1 }
      , i1_prev { i1_prev }
      , i2 { i2 }
      , i2_prev { i2_prev }
      , i3 { i3 }
      , i3_prev { i3_prev }
      , dx1 { dx1 }
      , dx1_prev { dx1_prev }
      , dx2 { dx2 }
      , dx2_prev { dx2_prev }
      , dx3 { dx3 }
      , dx3_prev { dx3_prev }
      , ux1 { ux1 }
      , ux2 { ux2 }
      , ux3 { ux3 }
      , weight { weight }
      , phi { phi }
      , pld_r { pld_r }
      , pld_i { pld_i }
      , tag { tag }
      , outgoing_indices { outgoing_indices } {}

    Inline void operator()(prtlidx_t p) const {
      npart_t idx;
      if (p >= npart_holes) {
        idx = npart + p - npart_holes;
      } else {
        idx = outgoing_indices(p);
      }
      if constexpr (D == Dim::_1D or D == Dim::_2D or D == Dim::_3D) {
        i1(idx)       = recv_buff_int(NINTS * p + 0);
        i1_prev(idx)  = recv_buff_int(NINTS * p + 1);
        dx1(idx)      = recv_buff_prtldx(NPRTLDX * p + 0);
        dx1_prev(idx) = recv_buff_prtldx(NPRTLDX * p + 1);
      }
      if constexpr (D == Dim::_2D or D == Dim::_3D) {
        i2(idx)       = recv_buff_int(NINTS * p + 2);
        i2_prev(idx)  = recv_buff_int(NINTS * p + 3);
        dx2(idx)      = recv_buff_prtldx(NPRTLDX * p + 2);
        dx2_prev(idx) = recv_buff_prtldx(NPRTLDX * p + 3);
      }
      if constexpr (D == Dim::_3D) {
        i3(idx)       = recv_buff_int(NINTS * p + 4);
        i3_prev(idx)  = recv_buff_int(NINTS * p + 5);
        dx3(idx)      = recv_buff_prtldx(NPRTLDX * p + 4);
        dx3_prev(idx) = recv_buff_prtldx(NPRTLDX * p + 5);
      }
      ux1(idx)    = recv_buff_real(NREALS * p + 0);
      ux2(idx)    = recv_buff_real(NREALS * p + 1);
      ux3(idx)    = recv_buff_real(NREALS * p + 2);
      weight(idx) = recv_buff_real(NREALS * p + 3);
      if constexpr (D == Dim::_2D and C != Coord::Cartesian) {
        phi(idx) = recv_buff_real(NREALS * p + 4);
      }
      if (NPLDS_R > 0) {
        for (auto l { 0u }; l < NPLDS_R; ++l) {
          pld_r(idx, l) = recv_buff_pld_r(NPLDS_R * p + l);
        }
      }
      if (NPLDS_I > 0) {
        for (auto l { 0u }; l < NPLDS_I; ++l) {
          pld_i(idx, l) = recv_buff_pld_i(NPLDS_I * p + l);
        }
      }
      tag(idx) = ParticleTag::alive;
    }
  };

} // namespace kernel::comm

#endif // KERNELS_COMM_HPP
