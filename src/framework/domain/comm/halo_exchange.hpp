/**
 * @file framework/domain/comm/halo_exchange.hpp
 * @brief Batched halo exchange: every field and every neighbor direction of
 *        one CommunicateFields / SynchronizeFields call is packed by one
 *        kernel, sent as one message per direction with all messages in
 *        flight at once, and unpacked by one kernel (fill) or by one kernel per
 *        direction in direction order (additive). The combined mode
 *        (SyncFill) sends the messages of the additive sync and computes the
 *        summed owned cells and the filled ghost cells in one kernel.
 * @implements
 *   - comm::HaloSegment
 *   - comm::HaloChunk
 *   - comm::HaloSlot
 *   - comm::HaloField
 *   - comm::HaloDirection
 *   - comm::MakeHaloField<> -> comm::HaloField
 *   - comm::HaloSyncFill_kernel<>
 *   - comm::HaloExchanger<>
 * @namespaces:
 *   - comm::
 * @macros:
 *   - DEVICE_ENABLED
 *   - GPU_AWARE_MPI
 * @note Used only when MPI_ENABLED is set.
 * @note Message and buffer layout: the message of one direction is the
 *       concatenation of the blocks of all fields of the call, in the field
 *       order given by the caller; a block lists its (i1, i2, i3, comp)
 *       elements with i1 fastest. Sender and receiver build the same layout
 *       from the same slice extents, so no metadata travels with the data.
 * @note Pack buffers, receive buffers, segment tables and the SynchronizeFields
 *       accumulators persist across calls (grow-only); a cached segment table
 *       is re-uploaded only when the slices or the field allocations change.
 */

#ifndef FRAMEWORK_DOMAIN_COMM_HALO_EXCHANGE_HPP
#define FRAMEWORK_DOMAIN_COMM_HALO_EXCHANGE_HPP

#include "global.h"

#include "arch/directions.h"
#include "arch/kokkos_aliases.h"
#include "arch/mpi_aliases.h"
#include "utils/error.h"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <map>
#include <utility>
#include <vector>

namespace comm {
  using namespace ntt;

  /**
   * One rectangular block [lo, lo + n) x [c0, c0 + nc) of a field, moved
   * between a field and a flat message buffer, or between two blocks of
   * fields with identical strides. All members are 64-bit so the struct has
   * no padding and two tables compare with memcmp.
   */
  struct HaloSegment {
    enum Kind : int64_t {
      Pack,      // buffer   <- src block
      Unpack,    // dst block  = buffer
      UnpackAdd, // dst block += buffer
      Copy,      // dst block  = src block
      CopyAdd    // dst block += src block
    };

    real_t* src { nullptr };
    real_t* dst { nullptr };
    int64_t st[4] { 0, 0, 0, 0 }; // element strides of (i1, i2, i3, comp)
    int64_t src_lo[3] { 0, 0, 0 };
    int64_t dst_lo[3] { 0, 0, 0 };
    int64_t n[3] { 1, 1, 1 }; // block extent per axis (1 for absent axes)
    int64_t c0 { 0 };
    int64_t nc { 0 };
    int64_t off { 0 }; // element offset of the block in its flat buffer
    int64_t kind { Pack };

    [[nodiscard]]
    auto size() const -> int64_t {
      return n[0] * n[1] * n[2] * nc;
    }
  };

  /**
   * A field taking part in an exchange: values are read from `src` and
   * written (fill) or accumulated (additive) into `dst`; src == dst for a
   * plain ghost fill. Both must share the strides `st`.
   */
  struct HaloField {
    real_t*      src { nullptr };
    real_t*      dst { nullptr };
    int64_t      st[4] { 0, 0, 0, 0 };
    cell_range_t comps { 0, 0 };
  };

  template <Dimension D, unsigned short N>
  auto MakeHaloField(const ndfield_t<D, N>& src,
                     const ndfield_t<D, N>& dst,
                     const cell_range_t&    comps) -> HaloField {
    HaloField f;
    f.src = src.data();
    f.dst = dst.data();
    for (auto r { 0u }; r <= static_cast<unsigned>(D); ++r) {
      raise::ErrorIf(src.stride(r) != dst.stride(r),
                     "HaloExchange: source and destination layouts differ",
                     HERE);
      raise::ErrorIf(src.extent(r) != dst.extent(r),
                     "HaloExchange: source and destination extents differ",
                     HERE);
    }
    for (auto d { 0u }; d < static_cast<unsigned>(D); ++d) {
      f.st[d] = static_cast<int64_t>(src.stride(d));
    }
    f.st[3] = static_cast<int64_t>(src.stride(static_cast<unsigned>(D)));
    f.comps = comps;
    return f;
  }

  /**
   * One neighbor direction of a call. `local` marks the periodic self-wrap
   * (the domain is its own neighbor): the block is copied within the domain
   * and no message is sent. `tag` is the index of the direction in
   * dir::Directions<D>::all, which the sender and the receiver of a message
   * share.
   */
  struct HaloDirection {
    int                       tag { 0 };
    bool                      local { false };
    int                       send_rank { -1 };
    int                       recv_rank { -1 };
    std::vector<cell_range_t> send_slice;
    std::vector<cell_range_t> recv_slice;
  };

  /**
   * Up to HaloSegments_kernel::CH consecutive elements of one segment: the
   * unit of work of one group of threads. A row chunk lies in one i1 row of
   * the block and starts at block-local (i1, i2, i3, c); the elements of a
   * flat chunk are located by index decomposition.
   */
  struct HaloChunk {
    int32_t seg { 0 };   // segment of the plan
    int32_t loc { 0 };   // element offset of the first element in the segment
    int32_t count { 0 }; // number of elements
    int32_t row { 0 };   // 1: row chunk
    int32_t i1 { 0 }, i2 { 0 }, i3 { 0 }, c { 0 };
  };

  /**
   * Device functor applying the chunks [k0, k0 + n) of a table: thread t
   * handles element t % CH of chunk k0 + t / CH. The CH threads of a chunk
   * read the same chunk and segment descriptors; along a row chunk the
   * element index needs no division.
   */
  struct HaloSegments_kernel {
    static constexpr int     LOG2_CH = 8;
    static constexpr int32_t CH      = 1 << LOG2_CH;

    Kokkos::View<const HaloSegment*> seg;
    Kokkos::View<const HaloChunk*>   chunk;
    Kokkos::View<real_t*>            sbuf;
    Kokkos::View<const real_t*>      rbuf;
    int64_t                          k0;

    Inline void operator()(const int64_t t) const {
      const HaloChunk& ch = chunk(k0 + (t >> LOG2_CH));
      const auto lane = static_cast<int32_t>(t & static_cast<int64_t>(CH - 1));
      if (lane >= ch.count) {
        return;
      }
      const HaloSegment& g   = seg(ch.seg);
      const int64_t      loc = static_cast<int64_t>(ch.loc) + lane;
      int64_t            i1, i2, i3, c;
      if (ch.row != 0) {
        i1 = ch.i1 + lane;
        i2 = ch.i2;
        i3 = ch.i3;
        c  = g.c0 + ch.c;
      } else {
        // block-local index decomposition in 32 bit (a halo block holds far
        // fewer than 2^32 elements; 64-bit integer division is emulated on
        // GPUs)
        const auto     n0 = static_cast<uint32_t>(g.n[0]);
        const auto     n1 = static_cast<uint32_t>(g.n[1]);
        const auto     n2 = static_cast<uint32_t>(g.n[2]);
        const auto     r  = static_cast<uint32_t>(loc);
        const uint32_t q1 = r / n0;
        const uint32_t q2 = q1 / n1;
        const uint32_t q3 = q2 / n2;
        i1                = r - q1 * n0;
        i2                = q1 - q2 * n1;
        i3                = q2 - q3 * n2;
        c                 = g.c0 + static_cast<int64_t>(q3);
      }
      const int64_t a_src = (g.src_lo[0] + i1) * g.st[0] +
                            (g.src_lo[1] + i2) * g.st[1] +
                            (g.src_lo[2] + i3) * g.st[2] + c * g.st[3];
      const int64_t a_dst = (g.dst_lo[0] + i1) * g.st[0] +
                            (g.dst_lo[1] + i2) * g.st[1] +
                            (g.dst_lo[2] + i3) * g.st[2] + c * g.st[3];
      if (g.kind == HaloSegment::Pack) {
        sbuf(g.off + loc) = g.src[a_src];
      } else if (g.kind == HaloSegment::Unpack) {
        g.dst[a_dst] = rbuf(g.off + loc);
      } else if (g.kind == HaloSegment::UnpackAdd) {
        g.dst[a_dst] += rbuf(g.off + loc);
      } else if (g.kind == HaloSegment::Copy) {
        g.dst[a_dst] = g.src[a_src];
      } else {
        g.dst[a_dst] += g.src[a_src];
      }
    }
  };

  /**
   * The values a SyncFill call receives from the neighbor behind one
   * direction e (the domain at offset -e): the block [lo, lo + n) of the
   * field they belong to and their offset in the receive buffer, or in the
   * send buffer for a periodic self-wrap (the domain's own values, packed
   * there). All members are 64-bit so the struct has no padding and two
   * tables compare with memcmp.
   */
  struct HaloSlot {
    int64_t valid { 0 };
    int64_t in_send { 0 };
    int64_t lo[3] { 0, 0, 0 };
    int64_t n[3] { 1, 1, 1 };
    int64_t off { 0 };
  };

  /**
   * Device functor of HaloExchanger::SyncFill (thread l: cell l of the field
   * storage, i1 fastest). The offset dl of a cell is -1 / +1 along an axis
   * below / above the owned box (on sides with a neighbor), else 0.
   *  - owned cell (dl = 0): its own value plus the values of the neighbors'
   *    blocks covering it, summed in direction order -- the additive sync;
   *  - ghost cell: the value the owner, the domain at offset dl, computes
   *    for it in the same way: the owner's own value (the block behind -dl)
   *    plus, for the owner's directions d in order, the value of the domain
   *    at offset -d from the owner, i.e. the block behind e = d - dl here
   *    (this domain's own value for e = 0).
   * The same values are summed in the same order, so every ghost cell is
   * bitwise equal to its owner's copy.
   */
  template <Dimension D>
  struct HaloSyncFill_kernel {
    Kokkos::View<const HaloSlot*> slot; // 27 entries, by direction code
    Kokkos::Array<int, 26>      order;  // codes of Directions<D>::all, in order
    int                         nord;
    Kokkos::View<const real_t*> sbuf, rbuf;
    real_t*                     fld;
    int64_t                     st[4];
    int64_t                     own_lo[3], own_hi[3];
    uint32_t                    ext[3];
    int                         c0, nc;

    // code of offset e: (e1 + 1) + 3 (e2 + 1) + 9 (e3 + 1)
    static Inline auto code(int e0, int e1, int e2) -> int {
      return (e0 + 1) + 3 * (e1 + 1) + 9 * (e2 + 1);
    }

    // value at cell x, comp c of the block behind direction code cd; false if
    // there is none or it does not cover x
    Inline auto value(int cd, const int64_t x[3], int c, real_t& v) const -> bool {
      const HaloSlot& s = slot(cd);
      if (s.valid == 0) {
        return false;
      }
      for (auto k { 0u }; k < 3u; ++k) {
        if (x[k] < s.lo[k] or x[k] >= s.lo[k] + s.n[k]) {
          return false;
        }
      }
      const int64_t idx = (x[0] - s.lo[0]) +
                          s.n[0] *
                            ((x[1] - s.lo[1]) +
                             s.n[1] * ((x[2] - s.lo[2]) + s.n[2] * (c - c0)));
      v = (s.in_send != 0) ? sbuf(s.off + idx) : rbuf(s.off + idx);
      return true;
    }

    Inline void operator()(const int64_t l) const {
      const auto     r = static_cast<uint32_t>(l);
      const uint32_t q = r / ext[0];
      const int64_t  x[3] { static_cast<int64_t>(r - q * ext[0]),
                            static_cast<int64_t>(q % ext[1]),
                            static_cast<int64_t>(q / ext[1]) };
      int            dl[3] { 0, 0, 0 };
      for (auto k { 0u }; k < 3u; ++k) {
        dl[k] = (x[k] < own_lo[k]) ? -1 : ((x[k] >= own_hi[k]) ? 1 : 0);
      }
      const bool    owned = (dl[0] == 0) and (dl[1] == 0) and (dl[2] == 0);
      const int64_t a     = x[0] * st[0] + x[1] * st[1] + x[2] * st[2];
      for (int c = c0; c < c0 + nc; ++c) {
        real_t acc = ZERO;
        bool   any = false;
        for (int k = 0; k < nord; ++k) {
          const int cd = order[k];
          const int e0 = cd % 3 - 1 - dl[0];
          const int e1 = (cd / 3) % 3 - 1 - dl[1];
          const int e2 = cd / 9 - 1 - dl[2];
          if (e0 < -1 or e0 > 1 or e1 < -1 or e1 > 1 or e2 < -1 or e2 > 1) {
            continue;
          }
          real_t v;
          if (e0 == 0 and e1 == 0 and e2 == 0) {
            v = fld[a + c * st[3]];
          } else if (not value(code(e0, e1, e2), x, c, v)) {
            continue;
          }
          acc += v;
          any  = true;
        }
        if (owned) {
          if (any) {
            fld[a + c * st[3]] += acc;
          }
        } else {
          real_t base;
          if (value(code(-dl[0], -dl[1], -dl[2]), x, c, base)) {
            fld[a + c * st[3]] = base + acc;
          }
        }
      }
    }
  };

  template <Dimension D>
  class HaloExchanger {
    using seg_range_t = std::pair<std::size_t, std::size_t>;

    struct Message {
      int     rank;
      int     tag;
      int64_t off;
      int64_t count;

      auto operator==(const Message& o) const -> bool {
        return (rank == o.rank) and (tag == o.tag) and (off == o.off) and
               (count == o.count);
      }
    };

    struct Plan {
      std::vector<HaloSegment> h_seg;
      std::vector<int64_t>     h_start; // prefix element counts, size nseg + 1
      seg_range_t              out { 0u, 0u };   // pack (+ fill-mode local copies)
      seg_range_t              in { 0u, 0u };    // fill-mode unpack
      std::vector<seg_range_t> per_dir;          // additive unpack, direction order
      std::vector<Message>     sends, recvs;
      int64_t                  send_total { 0 }, recv_total { 0 };
      std::vector<HaloSlot>  h_slot;  // SyncFill: 27 entries, by direction code
      std::vector<HaloChunk> h_chunk; // the segments cut into chunks
      std::vector<int64_t> h_chunk_start; // first chunk per segment, size nseg + 1

      Kokkos::View<HaloSegment*> d_seg;
      Kokkos::View<HaloChunk*>   d_chunk;
      Kokkos::View<HaloSlot*>    d_slot;

      auto same_as(const Plan& o) const -> bool {
        return (h_seg.size() == o.h_seg.size()) and
               (std::memcmp(h_seg.data(),
                            o.h_seg.data(),
                            h_seg.size() * sizeof(HaloSegment)) == 0) and
               (h_start == o.h_start) and (out == o.out) and (in == o.in) and
               (per_dir == o.per_dir) and (sends == o.sends) and
               (recvs == o.recvs) and (h_slot.size() == o.h_slot.size()) and
               (std::memcmp(h_slot.data(),
                            o.h_slot.data(),
                            h_slot.size() * sizeof(HaloSlot)) == 0);
      }
    };

    // message tags: base + direction index (unique per direction)
    static constexpr int TAG_BASE = 1024;
    // cached tables per call key; a key keeps up to MAX_VARIANTS tables (field
    // allocations swapped between calls, e.g. a ping-pong filter)
    static constexpr std::size_t MAX_VARIANTS = 4u;

    std::map<int, std::vector<Plan>> m_plans;
    std::map<int, std::size_t>       m_next_slot;

    Kokkos::View<real_t*> m_send, m_recv;
#if defined(DEVICE_ENABLED) && !defined(GPU_AWARE_MPI)
    Kokkos::View<real_t*, Kokkos::HostSpace> m_send_h, m_recv_h;
#endif
    std::vector<MPI_Request> m_reqs;

    std::map<int, ndfield_t<D, 6>> m_acc6;
    std::map<int, ndfield_t<D, 3>> m_acc3;

    static void set_block(int64_t* lo, int64_t* n, const std::vector<cell_range_t>& slice) {
      for (auto d { 0u }; d < static_cast<unsigned>(D); ++d) {
        lo[d] = static_cast<int64_t>(slice[d].first);
        n[d]  = static_cast<int64_t>(slice[d].second - slice[d].first);
      }
    }

    static auto segment(const HaloField&                 f,
                        HaloSegment::Kind                kind,
                        const std::vector<cell_range_t>* src_slice,
                        const std::vector<cell_range_t>* dst_slice,
                        int64_t                          off) -> HaloSegment {
      HaloSegment g;
      std::memset(&g, 0, sizeof(HaloSegment));
      g.n[0] = g.n[1] = g.n[2] = 1;
      g.src                    = f.src;
      g.dst                    = f.dst;
      for (auto r { 0u }; r < 4u; ++r) {
        g.st[r] = f.st[r];
      }
      if (src_slice != nullptr) {
        set_block(g.src_lo, g.n, *src_slice);
      }
      if (dst_slice != nullptr) {
        int64_t n_dst[3] { 1, 1, 1 };
        set_block(g.dst_lo, n_dst, *dst_slice);
        if (src_slice != nullptr) {
          for (auto d { 0u }; d < 3u; ++d) {
            raise::ErrorIf(n_dst[d] != g.n[d],
                           "HaloExchange: send and receive blocks differ",
                           HERE);
          }
        } else {
          for (auto d { 0u }; d < 3u; ++d) {
            g.n[d] = n_dst[d];
          }
        }
      }
      g.c0   = static_cast<int64_t>(f.comps.first);
      g.nc   = static_cast<int64_t>(f.comps.second - f.comps.first);
      g.off  = off;
      g.kind = kind;
      return g;
    }

    /**
     * Split every segment of `p` into chunks of at most CH elements: rows of
     * the block cut into chunks when the rows are long enough to fill them
     * (at least 3/4 of the threads busy), consecutive elements otherwise.
     */
    static void make_chunks(Plan& p) {
      constexpr int64_t CH = HaloSegments_kernel::CH;
      p.h_chunk.clear();
      p.h_chunk_start.assign(p.h_seg.size() + 1u, 0);
      for (auto s { 0u }; s < p.h_seg.size(); ++s) {
        const auto&   g     = p.h_seg[s];
        const int64_t n0    = g.n[0];
        const int64_t nrow  = (n0 + CH - 1) / CH;
        const bool    rows  = (4 * n0 >= 3 * nrow * CH);
        const int64_t total = g.size();
        raise::ErrorIf(total > static_cast<int64_t>(INT32_MAX),
                       "HaloExchange: block too large",
                       HERE);
        p.h_chunk_start[s] = static_cast<int64_t>(p.h_chunk.size());
        if (rows) {
          for (int64_t c = 0; c < g.nc; ++c) {
            for (int64_t i3 = 0; i3 < g.n[2]; ++i3) {
              for (int64_t i2 = 0; i2 < g.n[1]; ++i2) {
                const int64_t row0 = ((c * g.n[2] + i3) * g.n[1] + i2) * n0;
                for (int64_t i1 = 0; i1 < n0; i1 += CH) {
                  HaloChunk ch;
                  ch.seg   = static_cast<int32_t>(s);
                  ch.loc   = static_cast<int32_t>(row0 + i1);
                  ch.count = static_cast<int32_t>(std::min(CH, n0 - i1));
                  ch.row   = 1;
                  ch.i1    = static_cast<int32_t>(i1);
                  ch.i2    = static_cast<int32_t>(i2);
                  ch.i3    = static_cast<int32_t>(i3);
                  ch.c     = static_cast<int32_t>(c);
                  p.h_chunk.push_back(ch);
                }
              }
            }
          }
        } else {
          for (int64_t e = 0; e < total; e += CH) {
            HaloChunk ch;
            ch.seg   = static_cast<int32_t>(s);
            ch.loc   = static_cast<int32_t>(e);
            ch.count = static_cast<int32_t>(std::min(CH, total - e));
            p.h_chunk.push_back(ch);
          }
        }
      }
      p.h_chunk_start[p.h_seg.size()] = static_cast<int64_t>(p.h_chunk.size());
    }

    static auto build(const std::vector<HaloDirection>& dirs,
                      const std::vector<HaloField>&     flds,
                      bool                              additive) -> Plan {
      Plan p;
      // pack segments, one message per sending direction
      for (const auto& dr : dirs) {
        if (dr.local or dr.send_rank < 0) {
          continue;
        }
        const int64_t msg_off = p.send_total;
        for (const auto& f : flds) {
          p.h_seg.push_back(
            segment(f, HaloSegment::Pack, &dr.send_slice, nullptr, p.send_total));
          p.send_total += p.h_seg.back().size();
        }
        p.sends.push_back({ dr.send_rank, dr.tag, msg_off, p.send_total - msg_off });
      }
      // fill mode: the local wraps read active cells and write ghosts, so they
      // share the pack kernel
      if (not additive) {
        for (const auto& dr : dirs) {
          if (not dr.local) {
            continue;
          }
          for (const auto& f : flds) {
            p.h_seg.push_back(
              segment(f, HaloSegment::Copy, &dr.send_slice, &dr.recv_slice, 0));
          }
        }
      }
      p.out = { 0u, p.h_seg.size() };
      if (not additive) {
        // fill mode: receive blocks of different directions are disjoint, so
        // one kernel unpacks all of them
        for (const auto& dr : dirs) {
          if (dr.local or dr.recv_rank < 0) {
            continue;
          }
          const int64_t msg_off = p.recv_total;
          for (const auto& f : flds) {
            p.h_seg.push_back(
              segment(f, HaloSegment::Unpack, nullptr, &dr.recv_slice, p.recv_total));
            p.recv_total += p.h_seg.back().size();
          }
          p.recvs.push_back({ dr.recv_rank, dr.tag, msg_off, p.recv_total - msg_off });
        }
        p.in = { p.out.second, p.h_seg.size() };
      } else {
        // additive mode: receive blocks overlap, so contributions are summed
        // one direction at a time, in direction order
        for (const auto& dr : dirs) {
          const auto first = p.h_seg.size();
          if (dr.local) {
            for (const auto& f : flds) {
              p.h_seg.push_back(
                segment(f, HaloSegment::CopyAdd, &dr.send_slice, &dr.recv_slice, 0));
            }
          } else if (dr.recv_rank >= 0) {
            const int64_t msg_off = p.recv_total;
            for (const auto& f : flds) {
              p.h_seg.push_back(segment(f,
                                        HaloSegment::UnpackAdd,
                                        nullptr,
                                        &dr.recv_slice,
                                        p.recv_total));
              p.recv_total += p.h_seg.back().size();
            }
            p.recvs.push_back(
              { dr.recv_rank, dr.tag, msg_off, p.recv_total - msg_off });
          }
          if (p.h_seg.size() > first) {
            p.per_dir.push_back({ first, p.h_seg.size() });
          }
        }
      }
      p.h_start.resize(p.h_seg.size() + 1u, 0);
      for (auto s { 0u }; s < p.h_seg.size(); ++s) {
        p.h_start[s + 1] = p.h_start[s] + p.h_seg[s].size();
      }
      make_chunks(p);
      return p;
    }

    // code of the direction of Directions<D>::all with index `tag`
    static auto direction_code(int tag) -> int {
      const auto& d = dir::Directions<D>::all[static_cast<std::size_t>(tag)];
      int         e[3] { 0, 0, 0 };
      for (auto k { 0u }; k < static_cast<unsigned>(D); ++k) {
        e[k] = d[k];
      }
      return HaloSyncFill_kernel<D>::code(e[0], e[1], e[2]);
    }

    /**
     * SyncFill plan: the pack segments and messages of the additive sync of
     * `fld` (remote sends first; the self-wrap blocks packed after them, in
     * the send buffer), and the slot of every direction.
     */
    static auto build_sync_fill(const std::vector<HaloDirection>& dirs,
                                const HaloField&                  fld) -> Plan {
      Plan p;
      p.h_slot.resize(27u);
      for (auto& sl : p.h_slot) {
        std::memset(&sl, 0, sizeof(HaloSlot));
        sl.n[0] = sl.n[1] = sl.n[2] = 1;
      }
      for (const auto& dr : dirs) {
        if (dr.local or dr.send_rank < 0) {
          continue;
        }
        const int64_t msg_off = p.send_total;
        p.h_seg.push_back(
          segment(fld, HaloSegment::Pack, &dr.send_slice, nullptr, p.send_total));
        p.send_total += p.h_seg.back().size();
        p.sends.push_back({ dr.send_rank, dr.tag, msg_off, p.send_total - msg_off });
      }
      for (const auto& dr : dirs) {
        auto& sl = p.h_slot[static_cast<std::size_t>(direction_code(dr.tag))];
        if (dr.local) {
          // the values wrapped onto the receive block, in its element order
          const auto g = segment(fld,
                                 HaloSegment::Pack,
                                 &dr.send_slice,
                                 &dr.recv_slice,
                                 p.send_total);
          p.h_seg.push_back(g);
          sl.valid   = 1;
          sl.in_send = 1;
          sl.off     = p.send_total;
          set_block(sl.lo, sl.n, dr.recv_slice);
          p.send_total += g.size();
        } else if (dr.recv_rank >= 0) {
          HaloSegment g = segment(fld,
                                  HaloSegment::Unpack,
                                  nullptr,
                                  &dr.recv_slice,
                                  p.recv_total);
          sl.valid      = 1;
          sl.in_send    = 0;
          sl.off        = p.recv_total;
          set_block(sl.lo, sl.n, dr.recv_slice);
          p.recvs.push_back({ dr.recv_rank, dr.tag, p.recv_total, g.size() });
          p.recv_total += g.size();
        }
      }
      p.out = { 0u, p.h_seg.size() };
      p.h_start.resize(p.h_seg.size() + 1u, 0);
      for (auto s { 0u }; s < p.h_seg.size(); ++s) {
        p.h_start[s + 1] = p.h_start[s] + p.h_seg[s].size();
      }
      make_chunks(p);
      return p;
    }

    void upload(Plan& p) {
      const auto nseg   = p.h_seg.size();
      const auto nchunk = p.h_chunk.size();
      p.d_seg           = Kokkos::View<HaloSegment*> { "halo_segments", nseg };
      p.d_chunk         = Kokkos::View<HaloChunk*> { "halo_chunks", nchunk };
      auto h_seg        = Kokkos::create_mirror_view(p.d_seg);
      auto h_chnk       = Kokkos::create_mirror_view(p.d_chunk);
      for (auto s { 0u }; s < nseg; ++s) {
        h_seg(s) = p.h_seg[s];
      }
      for (auto k { 0u }; k < nchunk; ++k) {
        h_chnk(k) = p.h_chunk[k];
      }
      Kokkos::deep_copy(p.d_seg, h_seg);
      Kokkos::deep_copy(p.d_chunk, h_chnk);
      if (not p.h_slot.empty()) {
        p.d_slot   = Kokkos::View<HaloSlot*> { "halo_slots", p.h_slot.size() };
        auto h_slt = Kokkos::create_mirror_view(p.d_slot);
        for (auto s { 0u }; s < p.h_slot.size(); ++s) {
          h_slt(s) = p.h_slot[s];
        }
        Kokkos::deep_copy(p.d_slot, h_slt);
      }
    }

    // cached plan equal to `p` for `key`, uploading `p` if there is none
    auto cached(int key, Plan&& p) -> Plan& {
      auto& variants = m_plans[key];
      for (auto& v : variants) {
        if (v.same_as(p)) {
          return v;
        }
      }
      // tables of a replaced plan may still be read by queued kernels
      Kokkos::fence("HaloExchange: segment table update");
      upload(p);
      if (variants.size() < MAX_VARIANTS) {
        variants.push_back(std::move(p));
        return variants.back();
      }
      auto& slot = m_next_slot[key];
      auto& v    = variants[slot];
      slot       = (slot + 1u) % MAX_VARIANTS;
      v          = std::move(p);
      return v;
    }

    void ensure_buffers(int64_t nsend, int64_t nrecv) {
      if ((static_cast<int64_t>(m_send.extent(0)) < nsend) or
          (static_cast<int64_t>(m_recv.extent(0)) < nrecv)) {
        // buffers being replaced may still be read by queued kernels
        Kokkos::fence("HaloExchange: buffer resize");
        if (static_cast<int64_t>(m_send.extent(0)) < nsend) {
          m_send = Kokkos::View<real_t*> {
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "halo_send"),
            static_cast<std::size_t>(nsend)
          };
        }
        if (static_cast<int64_t>(m_recv.extent(0)) < nrecv) {
          m_recv = Kokkos::View<real_t*> {
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "halo_recv"),
            static_cast<std::size_t>(nrecv)
          };
        }
#if defined(DEVICE_ENABLED) && !defined(GPU_AWARE_MPI)
        m_send_h = Kokkos::create_mirror_view(m_send);
        m_recv_h = Kokkos::create_mirror_view(m_recv);
#endif
      }
    }

    void launch(const Plan& p, const seg_range_t& grp) {
      if (grp.second <= grp.first) {
        return;
      }
      const int64_t k0 = p.h_chunk_start[grp.first];
      const int64_t nk = p.h_chunk_start[grp.second] - k0;
      if (nk <= 0) {
        return;
      }
      Kokkos::parallel_for(
        "HaloExchange",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<int64_t>>(
          0,
          nk * HaloSegments_kernel::CH),
        HaloSegments_kernel { p.d_seg, p.d_chunk, m_send, m_recv, k0 });
    }

  public:
    HaloExchanger() = default;

    /**
     * @brief Exchange the blocks of `flds` over all directions of `dirs`.
     * @param key identifies the call site (tags + mode); its segment table is
     *        cached across calls
     * @param additive false: receive blocks overwrite dst (ghost fill);
     *        true: received and wrapped blocks are summed into dst one
     *        direction at a time, in the order of `dirs`
     */
    void Exchange(int                               key,
                  const std::vector<HaloDirection>& dirs,
                  const std::vector<HaloField>&     flds,
                  bool                              additive) {
      if (dirs.empty() or flds.empty()) {
        return;
      }
      auto& p = cached(key, build(dirs, flds, additive));
      ensure_buffers(p.send_total, p.recv_total);

      launch(p, p.out);
      communicate(p);

      if (not additive) {
        launch(p, p.in);
      } else {
        for (const auto& grp : p.per_dir) {
          launch(p, grp);
        }
      }
    }

    /**
     * @brief Additive halo sync of `fld` (deposit tails, src == dst) that
     *        also fills its ghost cells, in one exchange: owned cells get
     *        the sum of SynchronizeFields, ghost cells the bitwise copy of
     *        that sum on their owner (see HaloSyncFill_kernel).
     * @param dirs the synchronize-mode directions (active + ghost slices)
     * @param own_lo, own_hi the owned box per axis (ghosts included on the
     *        sides without a neighbor); 0 and 1 for absent axes
     * @param ext the storage extent per axis; 1 for absent axes
     */
    void SyncFill(int                               key,
                  const std::vector<HaloDirection>& dirs,
                  const HaloField&                  fld,
                  const int64_t                     own_lo[3],
                  const int64_t                     own_hi[3],
                  const int64_t                     ext[3]) {
      raise::ErrorIf(fld.src != fld.dst,
                     "HaloExchange: SyncFill acts on one field in place",
                     HERE);
      if (dirs.empty()) {
        return;
      }
      auto& p = cached(key, build_sync_fill(dirs, fld));
      ensure_buffers(p.send_total, p.recv_total);

      launch(p, p.out);
      communicate(p);

      HaloSyncFill_kernel<D> kern;
      kern.slot = p.d_slot;
      kern.nord = static_cast<int>(dir::Directions<D>::all.size());
      for (auto k { 0 }; k < kern.nord; ++k) {
        kern.order[k] = direction_code(k);
      }
      kern.sbuf = m_send;
      kern.rbuf = m_recv;
      kern.fld  = fld.dst;
      for (auto k { 0u }; k < 4u; ++k) {
        kern.st[k] = fld.st[k];
      }
      int64_t ncells = 1;
      for (auto k { 0u }; k < 3u; ++k) {
        kern.own_lo[k]  = own_lo[k];
        kern.own_hi[k]  = own_hi[k];
        kern.ext[k]     = static_cast<uint32_t>(ext[k]);
        ncells         *= ext[k];
      }
      raise::ErrorIf(ncells > static_cast<int64_t>(UINT32_MAX),
                     "HaloExchange: SyncFill field too large",
                     HERE);
      kern.c0 = static_cast<int>(fld.comps.first);
      kern.nc = static_cast<int>(fld.comps.second - fld.comps.first);
      Kokkos::parallel_for(
        "HaloSyncFill",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<int64_t>>(
          0,
          ncells),
        kern);
    }

  private:
    // the messages of a plan, all in flight at once
    void communicate(const Plan& p) {
      const bool remote = (not p.sends.empty()) or (not p.recvs.empty());
      if (remote) {
#if defined(DEVICE_ENABLED)
        // Intel GPUs: every kernel writing a send buffer must complete
        // before MPI reads it
        Kokkos::fence("HaloExchange: before MPI");
#endif
#if defined(DEVICE_ENABLED) && !defined(GPU_AWARE_MPI)
        if (p.send_total > 0) {
          const auto rng = std::make_pair(static_cast<std::size_t>(0),
                                          static_cast<std::size_t>(p.send_total));
          Kokkos::deep_copy(Kokkos::subview(m_send_h, rng),
                            Kokkos::subview(m_send, rng));
        }
        real_t* const sptr = m_send_h.data();
        real_t* const rptr = m_recv_h.data();
#else
        real_t* const sptr = m_send.data();
        real_t* const rptr = m_recv.data();
#endif
        m_reqs.assign(p.sends.size() + p.recvs.size(), MPI_REQUEST_NULL);
        std::size_t q = 0u;
        for (const auto& m : p.recvs) {
          raise::ErrorIf(m.count > static_cast<int64_t>(INT_MAX),
                         "HaloExchange: message too large",
                         HERE);
          MPI_Irecv(rptr + m.off,
                    static_cast<int>(m.count),
                    mpi::get_type<real_t>(),
                    m.rank,
                    TAG_BASE + m.tag,
                    MPI_COMM_WORLD,
                    &m_reqs[q++]);
        }
        for (const auto& m : p.sends) {
          raise::ErrorIf(m.count > static_cast<int64_t>(INT_MAX),
                         "HaloExchange: message too large",
                         HERE);
          MPI_Isend(sptr + m.off,
                    static_cast<int>(m.count),
                    mpi::get_type<real_t>(),
                    m.rank,
                    TAG_BASE + m.tag,
                    MPI_COMM_WORLD,
                    &m_reqs[q++]);
        }
        MPI_Waitall(static_cast<int>(m_reqs.size()), m_reqs.data(), MPI_STATUSES_IGNORE);
#if defined(DEVICE_ENABLED) && !defined(GPU_AWARE_MPI)
        if (p.recv_total > 0) {
          const auto rng = std::make_pair(static_cast<std::size_t>(0),
                                          static_cast<std::size_t>(p.recv_total));
          Kokkos::deep_copy(Kokkos::subview(m_recv, rng),
                            Kokkos::subview(m_recv_h, rng));
        }
#endif
      }
    }

  public:
    /**
     * @brief Persistent zeroed accumulator shaped like `like` (zeroing is
     *        asynchronous; the buffer is reallocated when the extents change).
     */
    template <unsigned short N>
    auto Accumulator(int slot, const ndfield_t<D, N>& like) -> ndfield_t<D, N> {
      auto& store = [this]() -> std::map<int, ndfield_t<D, N>>& {
        if constexpr (N == 6) {
          return m_acc6;
        } else {
          static_assert(N == 3, "HaloExchange: accumulators are 3- or 6-component");
          return m_acc3;
        }
      }();
      auto& acc     = store[slot];
      bool  matches = (acc.data() != nullptr);
      for (auto r { 0u }; matches and r < static_cast<unsigned>(D); ++r) {
        matches = (acc.extent(r) == like.extent(r));
      }
      if (not matches) {
        Kokkos::fence("HaloExchange: accumulator resize");
        if constexpr (D == Dim::_1D) {
          acc = ndfield_t<D, N> { "halo_accumulator", like.extent(0) };
        } else if constexpr (D == Dim::_2D) {
          acc = ndfield_t<D, N> { "halo_accumulator", like.extent(0), like.extent(1) };
        } else {
          acc = ndfield_t<D, N> { "halo_accumulator",
                                  like.extent(0),
                                  like.extent(1),
                                  like.extent(2) };
        }
      } else {
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace {}, acc, ZERO);
      }
      return acc;
    }
  };

} // namespace comm

#endif // FRAMEWORK_DOMAIN_COMM_HALO_EXCHANGE_HPP
