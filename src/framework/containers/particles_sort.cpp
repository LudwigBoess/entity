#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/sorting.h"

#include "framework/containers/particles.h"
#include "framework/domain/grid.h"

#if defined(TILED_DEPOSIT)
  #if (defined(SYCL_ENABLED) && defined(ONEDPL_ENABLED)) ||                    \
    (defined(CUDA_ENABLED) && defined(THRUST_ENABLED)) ||                      \
    (defined(HIP_ENABLED) && defined(ROCTHRUST_ENABLED))
    #define TILED_DEPOSIT_USE_VENDOR_SORT
    #include "utils/sort_dispatch.h"
  #endif
#endif

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <Kokkos_Sort.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace ntt {

  template <typename T>
  void RemoveDeadInArray(array_t<T*>& arr, const array_t<npart_t*>& indices_alive) {
    const npart_t n_alive = indices_alive.extent(0);
    auto          buffer  = Kokkos::View<T*>("buffer", n_alive);
    Kokkos::parallel_for(
      "PopulateBufferAlive",
      n_alive,
      Lambda(prtlidx_t p) { buffer(p) = arr(indices_alive(p)); });

    Kokkos::deep_copy(
      Kokkos::subview(arr, std::make_pair(static_cast<npart_t>(0), n_alive)),
      buffer);
  }

  template <typename T>
  void RemoveDeadInArray(array_t<T**>& arr, const array_t<npart_t*>& indices_alive) {
    const npart_t n_alive = indices_alive.extent(0);
    auto          buffer  = array_t<T**> { "buffer", n_alive, arr.extent(1) };
    Kokkos::parallel_for(
      "PopulateBufferAlive",
      CreateParticleRangePolicy<Dim::_2D>(
        { 0, 0 },
        { n_alive, static_cast<npart_t>(arr.extent(1)) }),
      Lambda(prtlidx_t p, prtlidx_t l) {
        buffer(p, l) = arr(indices_alive(p), l);
      });

    Kokkos::deep_copy(
      Kokkos::subview(arr,
                      std::make_pair(static_cast<npart_t>(0), n_alive),
                      Kokkos::ALL),
      buffer);
  }

  template <Dimension D, Coord::type C>
  void Particles<D, C>::RemoveDead() {
    npart_t n_alive = 0, n_dead = 0;
    auto&   this_tag = tag;

    Kokkos::parallel_reduce(
      "CountDeadAlive",
      rangeActiveParticles(),
      Lambda(prtlidx_t p, npart_t & nalive, npart_t & ndead) {
        nalive += (this_tag(p) == ParticleTag::alive);
        ndead  += (this_tag(p) == ParticleTag::dead);
        if (this_tag(p) != ParticleTag::alive and this_tag(p) != ParticleTag::dead) {
          raise::KernelError(HERE, "wrong particle tag");
        }
      },
      n_alive,
      n_dead);

    const array_t<npart_t*> indices_alive { "indices_alive", n_alive };
    const array_t<npart_t*> alive_counter { "counter_alive", 1 };

    Kokkos::parallel_for(
      "AliveIndices",
      rangeActiveParticles(),
      Lambda(prtlidx_t p) {
        if (this_tag(p) == ParticleTag::alive) {
          const auto idx     = Kokkos::atomic_fetch_add(&alive_counter(0), 1);
          indices_alive(idx) = p;
        }
      });

    {
      auto alive_counter_h = Kokkos::create_mirror_view(alive_counter);
      Kokkos::deep_copy(alive_counter_h, alive_counter);
      raise::ErrorIf(alive_counter_h(0) != n_alive,
                     "error in finding alive particle indices",
                     HERE);
    }

    if constexpr (D == Dim::_1D or D == Dim::_2D or D == Dim::_3D) {
      RemoveDeadInArray(i1, indices_alive);
      RemoveDeadInArray(i1_prev, indices_alive);
      RemoveDeadInArray(dx1, indices_alive);
      RemoveDeadInArray(dx1_prev, indices_alive);
    }

    if constexpr (D == Dim::_2D or D == Dim::_3D) {
      RemoveDeadInArray(i2, indices_alive);
      RemoveDeadInArray(i2_prev, indices_alive);
      RemoveDeadInArray(dx2, indices_alive);
      RemoveDeadInArray(dx2_prev, indices_alive);
    }

    if constexpr (D == Dim::_3D) {
      RemoveDeadInArray(i3, indices_alive);
      RemoveDeadInArray(i3_prev, indices_alive);
      RemoveDeadInArray(dx3, indices_alive);
      RemoveDeadInArray(dx3_prev, indices_alive);
    }

    RemoveDeadInArray(ux1, indices_alive);
    RemoveDeadInArray(ux2, indices_alive);
    RemoveDeadInArray(ux3, indices_alive);
    RemoveDeadInArray(weight, indices_alive);

    if constexpr (D == Dim::_2D && C != Coord::Cartesian) {
      RemoveDeadInArray(phi, indices_alive);
    }

    if (npld_r() > 0) {
      RemoveDeadInArray(pld_r, indices_alive);
    }

    if (npld_i() > 0) {
      RemoveDeadInArray(pld_i, indices_alive);
    }

    Kokkos::Experimental::fill(
      "TagAliveParticles",
      Kokkos::DefaultExecutionSpace(),
      Kokkos::subview(this_tag, std::make_pair(static_cast<npart_t>(0), n_alive)),
      ParticleTag::alive);

    Kokkos::Experimental::fill(
      "TagDeadParticles",
      Kokkos::DefaultExecutionSpace(),
      Kokkos::subview(this_tag, std::make_pair(n_alive, n_alive + n_dead)),
      ParticleTag::dead);

    set_npart(n_alive);
    m_is_sorted = true;
  }

  namespace {
    // Grow-only reserve for a persistent sort-scratch View: (re)allocate to
    // `n` (plus 1/8 headroom) only when the current capacity is smaller, so a
    // slowly growing npart triggers a handful of reallocations over a run
    // instead of one device alloc/free per sort. The old buffer is released
    // before the new one is allocated, keeping the transient peak at a single
    // buffer during the (rare) grow. Callers always operate on the leading
    // `[0, n)` prefix, so the extra headroom slots are never read.
    template <typename V>
    inline void reserve_scratch_1d(V& v, const char* label, npart_t n) {
      if (static_cast<npart_t>(v.extent(0)) < n) {
        v = V {};
        v = V { label, n + n / 8u };
      }
    }

    template <typename V>
    inline void reserve_scratch_2d(V& v, const char* label, npart_t n, npart_t ncols) {
      if (static_cast<npart_t>(v.extent(0)) < n or
          static_cast<npart_t>(v.extent(1)) != ncols) {
        v = V {};
        v = V { label, n + n / 8u, ncols };
      }
    }

#if defined(TILED_DEPOSIT) && defined(TILED_DEPOSIT_USE_VENDOR_SORT) &&        \
  defined(SYCL_ENABLED) && defined(ONEDPL_ENABLED)
    // sort_by_key of the first n tile keys through a copy in the narrower
    // unsigned type K (persistent `narrow` buffer, left sorted for the tile
    // offsets)
    template <typename K>
    void sort_narrow_keys(const array_t<ncells_t*>& keys,
                          array_t<K*>&              narrow,
                          const char*               label,
                          prtl_perm_t&              perm,
                          ncells_t                  n_bins,
                          npart_t                   n) {
      reserve_scratch_1d(narrow, label, n);
      auto k32 = keys;
      auto kn  = narrow;
      Kokkos::parallel_for(
        "NarrowTileKeys",
        n,
        KOKKOS_LAMBDA(const npart_t p) { kn(p) = static_cast<K>(k32(p)); });
      sort_helpers::sort_by_key_dispatch(narrow,
                                         perm,
                                         n_bins,
                                         n,
                                         sort::backend::OneDPL {});
    }
#endif
  } // namespace

#if defined(TILED_DEPOSIT)
  template <Dimension D, Coord::type C>
  template <typename K>
  void Particles<D, C>::compute_tile_offsets(const array_t<K*>& tile_indices,
                                             ncells_t           total_tiles,
                                             npart_t            npart_local) {
    // Per-tile prefix sum `tile_offsets` for the tiled pusher from the
    // (already sorted) `tile_indices`: non-decreasing for alive particles,
    // with the dead sentinel `total_tiles + 1` clustered at the end. Tile t
    // starts at the first particle whose key is >= t (binary search, one
    // thread per tile), so an empty tile starts where the next non-empty one
    // does, and tile_offsets(total_tiles) is the first dead particle, i.e.
    // the alive count. The buffer persists across sorts (same tile count).
    if (m_tile_layout.tile_offsets.extent(0) != total_tiles + 1u) {
      m_tile_layout.tile_offsets = array_t<npart_t*> { "tile_offsets",
                                                       total_tiles + 1u };
    }
    auto offsets = m_tile_layout.tile_offsets;
    auto keys    = tile_indices;
    Kokkos::parallel_for(
      "TileOffsets",
      CreateParticleRangePolicy<Dim::_1D>({ 0u }, { total_tiles + 1u }),
      Lambda(prtlidx_t t) {
        npart_t lo = 0u, hi = npart_local;
        while (lo < hi) {
          const npart_t mid = lo + (hi - lo) / 2u;
          if (static_cast<ncells_t>(keys(mid)) < static_cast<ncells_t>(t)) {
            lo = mid + 1u;
          } else {
            hi = mid;
          }
        }
        offsets(t) = lo;
      });

    // tile_offsets(total_tiles) is the alive-particle count at sort time:
    // the tiles partition exactly [0, npart_partitioned). The deposit
    // launcher compares this against the live npart() to detect (and
    // separately deposit) particles appended since this sort.
    npart_t n_alive { 0u };
    Kokkos::deep_copy(n_alive, Kokkos::subview(offsets, total_tiles));
    m_tile_layout.npart_partitioned = n_alive;
  }
#endif // TILED_DEPOSIT

  template <Dimension D, Coord::type C>
  void Particles<D, C>::SortSpatially(const Grid<D>& grid) {
#if defined(TILED_DEPOSIT)
    // ---------------------- tiled_deposit: tile-based sort ------------------ //
    const auto npart_local = npart();
    if (npart_local == 0u) {
      m_tile_layout = TileLayout<D> {};
      m_is_sorted   = true;
      return;
    }

    constexpr unsigned short T = static_cast<unsigned short>(
      TILED_DEPOSIT_TILE_SIZE);
    static_assert(T > 0u, "TILED_DEPOSIT_TILE_SIZE must be > 0");

    // 1. Compute per-axis tile counts and total_tiles.
    const auto ncells_active = grid.n_active();
    ncells_t   ntx[3] { 1u, 1u, 1u };
    ncells_t   total_tiles { 1u };
    if constexpr ((D == Dim::_1D) or (D == Dim::_2D) or (D == Dim::_3D)) {
      ntx[0]       = static_cast<ncells_t>(math::ceil(
        static_cast<double>(ncells_active[0]) / static_cast<double>(T)));
      total_tiles *= ntx[0];
    }
    if constexpr ((D == Dim::_2D) or (D == Dim::_3D)) {
      ntx[1]       = static_cast<ncells_t>(math::ceil(
        static_cast<double>(ncells_active[1]) / static_cast<double>(T)));
      total_tiles *= ntx[1];
    }
    if constexpr (D == Dim::_3D) {
      ntx[2]       = static_cast<ncells_t>(math::ceil(
        static_cast<double>(ncells_active[2]) / static_cast<double>(T)));
      total_tiles *= ntx[2];
    }

    // 2. Compute per-particle tile key (with min(i, i_prev)).
  #if defined(TILED_DEPOSIT_USE_VENDOR_SORT) && defined(SYCL_ENABLED) &&       \
    defined(ONEDPL_ENABLED)
    // oneDPL sorts the keys in place, so reuse a persistent, grow-only keys
    // buffer instead of allocating a fresh one every sort. `tile_indices`
    // aliases the (possibly over-capacity) persistent buffer; downstream
    // consumers bound their work by the explicit `npart_local`, never by
    // `tile_indices.extent(0)`.
    reserve_scratch_1d(m_sort_keys, "tile_indices", npart_local);
    array_t<ncells_t*> tile_indices = m_sort_keys;
  #else
    array_t<ncells_t*> tile_indices { "tile_indices", npart_local };
  #endif
    Kokkos::parallel_for(
      "FillTileIndices",
      rangeActiveParticles(),
      sort::PositionToTileIndex<D, false, true> { i1,
                                                  i2,
                                                  i3,
                                                  tag,
                                                  tile_indices,
                                                  ncells_active,
                                                  static_cast<ncells_t>(T),
                                                  array_t<npart_t*> {},
                                                  i1_prev,
                                                  i2_prev,
                                                  i3_prev });

    // 3. Sort. Vendor library (oneDPL/Thrust) when compiled in;
    //    Kokkos::BinSort otherwise. n_bins = total_tiles + 2 covers
    //    the dead-particle sentinel bin (total_tiles + 1u).
    const ncells_t n_bins = total_tiles + 2u;
    const auto     slice  = prtl_slice_t(0, npart_local);

  #if defined(TILED_DEPOSIT_USE_VENDOR_SORT)
    // Vendor path: produce an explicit permutation via sort_by_key, then
    // apply it to the SoA members by gathering the alive prefix into a
    // scratch buffer per member (one kernel) and copying back in place (one
    // kernel). The *_prev arrays are skipped — see apply_permutation_to_soa.
    // On the SYCL/oneDPL path the keys, perm, and gather scratch are all
    // persistent, grow-only buffers (class members) reused across sorts, so
    // steady state makes no per-sort device allocation; the fixed cost is
    // the `npart_partitioned`-sized buffers held resident.
    #if defined(SYCL_ENABLED) && defined(ONEDPL_ENABLED)
    // Persistent, reused perm: oneDPL's sort_by_key is in place and never
    // reassigns `perm`, so it can share a grow-only buffer across sorts (no
    // per-sort device allocation). `npart_local` is passed explicitly since
    // `tile_indices` is now an over-capacity persistent buffer whose extent
    // is the reserved capacity, not the count to sort.
    reserve_scratch_1d(m_sort_perm, "tile_perm", npart_local);
    prtl_perm_t perm = m_sort_perm;
    // sort the keys in the narrowest unsigned type that holds n_bins (the
    // radix sort makes one pass per 4 key bits); the sort is stable, so the
    // permutation is that of the full-width keys. The tile offsets are built
    // from the sorted keys of the type sorted.
    if (n_bins <= 256u) {
      sort_narrow_keys(tile_indices, m_sort_keys8, "tile_indices8", perm, n_bins, npart_local);
      compute_tile_offsets(m_sort_keys8, total_tiles, npart_local);
    } else if (n_bins <= 65536u) {
      sort_narrow_keys(tile_indices,
                       m_sort_keys16,
                       "tile_indices16",
                       perm,
                       n_bins,
                       npart_local);
      compute_tile_offsets(m_sort_keys16, total_tiles, npart_local);
    } else {
      sort_helpers::sort_by_key_dispatch(tile_indices,
                                         perm,
                                         n_bins,
                                         npart_local,
                                         sort::backend::OneDPL {});
      compute_tile_offsets(tile_indices, total_tiles, npart_local);
    }
    #elif defined(HIP_ENABLED) && defined(ROCTHRUST_ENABLED)
    prtl_perm_t perm { "tile_perm", npart_local };
    sort_helpers::sort_by_key_dispatch(tile_indices,
                                       perm,
                                       n_bins,
                                       sort::backend::Rocthrust {});
    compute_tile_offsets(tile_indices, total_tiles, npart_local);
    #else
    prtl_perm_t perm { "tile_perm", npart_local };
    sort_helpers::sort_by_key_dispatch(tile_indices,
                                       perm,
                                       n_bins,
                                       sort::backend::Thrust {});
    compute_tile_offsets(tile_indices, total_tiles, npart_local);
    #endif
    // `tile_indices` is sorted in place by sort_by_key and no longer needed.
    // On the CUDA/HIP vendor path dropping the handle frees the transient
    // keys buffer before the gather; on SYCL `tile_indices` aliases the
    // persistent `m_sort_keys`, so this only drops the local handle (the
    // buffer is retained and reused next sort). The copy of the alive count
    // in compute_tile_offsets has waited for the sort.
    tile_indices = array_t<ncells_t*> {};
    // Gather only the alive particles. The sort binned dead particles to
    // the sentinel tile, so they occupy [npart_partitioned, npart_local)
    // and `perm[0, npart_partitioned)` lists the alive slots in tile
    // order. Restricting the gather to the alive count drops the dead in
    // the same pass (no separate compaction), skips work on dead slots,
    // and sizes the gather scratch to the alive count. The dead tail is
    // released below via `set_npart`.
    apply_permutation_to_soa(perm, m_tile_layout.npart_partitioned);
  #else
    // BinSort path: same mechanism as legacy SortSpatially (BinSort
    // allocates one temp View per `sorter.sort(view)` call and frees
    // it before the next), so peak transient memory is bounded.
    using sorter_op_t = Kokkos::BinOp1D<array_t<ncells_t*>>;
    using sorter_t    = Kokkos::BinSort<array_t<ncells_t*>, sorter_op_t>;
    auto bin_op       = sorter_op_t { static_cast<int>(n_bins), 0u, n_bins };
    auto sorter       = sorter_t { tile_indices, bin_op, false };
    sorter.create_permute_vector();
    if constexpr (D == Dim::_1D or D == Dim::_2D or D == Dim::_3D) {
      sorter.sort(Kokkos::subview(i1, slice));
      sorter.sort(Kokkos::subview(i1_prev, slice));
      sorter.sort(Kokkos::subview(dx1, slice));
      sorter.sort(Kokkos::subview(dx1_prev, slice));
    }
    if constexpr (D == Dim::_2D or D == Dim::_3D) {
      sorter.sort(Kokkos::subview(i2, slice));
      sorter.sort(Kokkos::subview(i2_prev, slice));
      sorter.sort(Kokkos::subview(dx2, slice));
      sorter.sort(Kokkos::subview(dx2_prev, slice));
    }
    if constexpr (D == Dim::_3D) {
      sorter.sort(Kokkos::subview(i3, slice));
      sorter.sort(Kokkos::subview(i3_prev, slice));
      sorter.sort(Kokkos::subview(dx3, slice));
      sorter.sort(Kokkos::subview(dx3_prev, slice));
    }
    sorter.sort(Kokkos::subview(ux1, slice));
    sorter.sort(Kokkos::subview(ux2, slice));
    sorter.sort(Kokkos::subview(ux3, slice));
    sorter.sort(Kokkos::subview(weight, slice));
    sorter.sort(Kokkos::subview(tag, slice));
    if constexpr (D == Dim::_2D and C != Coord::Cartesian) {
      sorter.sort(Kokkos::subview(phi, slice));
    }
    for (auto pldr { 0u }; pldr < npld_r(); ++pldr) {
      sorter.sort(Kokkos::subview(pld_r, slice, pldr));
    }
    for (auto pldi { 0u }; pldi < npld_i(); ++pldi) {
      sorter.sort(Kokkos::subview(pld_i, slice, pldi));
    }
    // Apply the same permutation to `tile_indices` itself so it ends
    // monotonically non-decreasing for the offsets pass, then build the
    // tile offsets from it (the in-place BinSort path has no separate
    // gather to hoist this ahead of).
    sorter.sort(tile_indices);
    compute_tile_offsets(tile_indices, total_tiles, npart_local);
  #endif // TILED_DEPOSIT_USE_VENDOR_SORT

    // Populate `m_tile_layout` size/shape. `tile_perm` is not used in the
    // current design — the SoA arrays are physically permuted into tile
    // order, so consumers iterate `[tile_offsets(t), tile_offsets(t+1))`
    // directly without a separate permutation indirection.
    m_tile_layout.ntiles_per_axis[0] = ntx[0];
    m_tile_layout.ntiles_per_axis[1] = ntx[1];
    m_tile_layout.ntiles_per_axis[2] = ntx[2];
    m_tile_layout.ntiles_total       = total_tiles;
    m_tile_layout.tile_size          = T;
    m_tile_layout.tile_perm          = prtl_perm_t {};
    m_is_sorted                      = true;

    // Compact-on-sort: drop the dead tail now instead of waiting for the
    // periodic RemoveDead. The sort parked dead particles in the sentinel
    // bin at [npart_partitioned, npart()), and the alive set is exactly
    // [0, npart_partitioned) — gathered into tile order above (vendor) or
    // sorted in place (BinSort). Shrinking npart() here keeps it, and
    // every subsequent sort's transient buffers, tracking the alive count
    // instead of ratcheting up with dead slots between clearing intervals.
    // (RemoveDead remains the compactor when spatial sorting is disabled.)
    set_npart(m_tile_layout.npart_partitioned);

    Kokkos::fence("SortSpatially: end of tiled_deposit path");
#else  // !TILED_DEPOSIT — legacy in-place BinSort by global cell index
    const auto nx2         = grid.n_active(in::x2);
    const auto nx3         = grid.n_active(in::x3);
    const auto total_cells = grid.num_active();

    array_t<ncells_t*> cell_indices { "cell_indices", npart() };

    Kokkos::parallel_for("FillCellIndices",
                         rangeActiveParticles(),
                         sort::PositionToTileIndex<D, false> { i1,
                                                               i2,
                                                               i3,
                                                               tag,
                                                               cell_indices,
                                                               grid.n_active() });
    const auto slice = prtl_slice_t(0, npart());

    using sorter_op_t = Kokkos::BinOp1D<decltype(cell_indices)>;
    using sorter_t    = Kokkos::BinSort<decltype(cell_indices), sorter_op_t>;
    auto bin_op       = sorter_op_t { static_cast<int>(total_cells + 1u),
                                0u,
                                total_cells + 1u };
    auto sorter       = sorter_t { cell_indices, bin_op, false };
    sorter.create_permute_vector();
    if constexpr (D == Dim::_1D or D == Dim::_2D or D == Dim::_3D) {
      sorter.sort(Kokkos::subview(i1, slice));
      sorter.sort(Kokkos::subview(i1_prev, slice));
      sorter.sort(Kokkos::subview(dx1, slice));
      sorter.sort(Kokkos::subview(dx1_prev, slice));
    }
    if constexpr (D == Dim::_2D or D == Dim::_3D) {
      sorter.sort(Kokkos::subview(i2, slice));
      sorter.sort(Kokkos::subview(i2_prev, slice));
      sorter.sort(Kokkos::subview(dx2, slice));
      sorter.sort(Kokkos::subview(dx2_prev, slice));
    }
    if constexpr (D == Dim::_3D) {
      sorter.sort(Kokkos::subview(i3, slice));
      sorter.sort(Kokkos::subview(i3_prev, slice));
      sorter.sort(Kokkos::subview(dx3, slice));
      sorter.sort(Kokkos::subview(dx3_prev, slice));
    }
    sorter.sort(Kokkos::subview(ux1, slice));
    sorter.sort(Kokkos::subview(ux2, slice));
    sorter.sort(Kokkos::subview(ux3, slice));
    sorter.sort(Kokkos::subview(weight, slice));
    sorter.sort(Kokkos::subview(tag, slice));
    if constexpr (D == Dim::_2D and C != Coord::Cartesian) {
      sorter.sort(Kokkos::subview(phi, slice));
    }
    for (auto pldr { 0u }; pldr < npld_r(); ++pldr) {
      sorter.sort(Kokkos::subview(pld_r, slice, pldr));
    }
    for (auto pldi { 0u }; pldi < npld_i(); ++pldi) {
      sorter.sort(Kokkos::subview(pld_i, slice, pldi));
    }
#endif // TILED_DEPOSIT
  }

#if defined(TILED_DEPOSIT_USE_VENDOR_SORT)
  namespace permute_helpers {

    /**
     * @brief The permutation of the particle SoA members in two passes over
     *        the first `n` particles: BACK = false gathers every member
     *        through `perm` into its scratch, BACK = true copies the scratch
     *        back into the members (each member keeps its storage). The
     *        *_prev arrays are not permuted; phi only in 2D non-Cartesian.
     */
    template <Dimension D, Coord::type C, bool BACK>
    struct PermuteSoA_kernel {
      ParticleArrays     a;
      array_t<int*>      si[3];
      array_t<prtldx_t*> sdx[3];
      array_t<real_t*>   su[5];
      array_t<short*>    stag;
      array_t<real_t**>  spr;
      array_t<npart_t**> spi;
      prtl_perm_t        perm;
      npart_t            npr, npi;

      Inline void operator()(const npart_t p) const {
        if constexpr (BACK) {
          if constexpr (D == Dim::_1D or D == Dim::_2D or D == Dim::_3D) {
            a.i1(p)  = si[0](p);
            a.dx1(p) = sdx[0](p);
          }
          if constexpr (D == Dim::_2D or D == Dim::_3D) {
            a.i2(p)  = si[1](p);
            a.dx2(p) = sdx[1](p);
          }
          if constexpr (D == Dim::_3D) {
            a.i3(p)  = si[2](p);
            a.dx3(p) = sdx[2](p);
          }
          a.ux1(p)    = su[0](p);
          a.ux2(p)    = su[1](p);
          a.ux3(p)    = su[2](p);
          a.weight(p) = su[3](p);
          if constexpr (D == Dim::_2D and C != Coord::Cartesian) {
            a.phi(p) = su[4](p);
          }
          a.tag(p) = stag(p);
          for (npart_t l = 0; l < npr; ++l) {
            a.pld_r(p, l) = spr(p, l);
          }
          for (npart_t l = 0; l < npi; ++l) {
            a.pld_i(p, l) = spi(p, l);
          }
        } else {
          const npart_t q = perm(p);
          if constexpr (D == Dim::_1D or D == Dim::_2D or D == Dim::_3D) {
            si[0](p)  = a.i1(q);
            sdx[0](p) = a.dx1(q);
          }
          if constexpr (D == Dim::_2D or D == Dim::_3D) {
            si[1](p)  = a.i2(q);
            sdx[1](p) = a.dx2(q);
          }
          if constexpr (D == Dim::_3D) {
            si[2](p)  = a.i3(q);
            sdx[2](p) = a.dx3(q);
          }
          su[0](p) = a.ux1(q);
          su[1](p) = a.ux2(q);
          su[2](p) = a.ux3(q);
          su[3](p) = a.weight(q);
          if constexpr (D == Dim::_2D and C != Coord::Cartesian) {
            su[4](p) = a.phi(q);
          }
          stag(p) = a.tag(q);
          for (npart_t l = 0; l < npr; ++l) {
            spr(p, l) = a.pld_r(q, l);
          }
          for (npart_t l = 0; l < npi; ++l) {
            spi(p, l) = a.pld_i(q, l);
          }
        }
      }
    };

  } // namespace permute_helpers

  template <Dimension D, Coord::type C>
  void Particles<D, C>::apply_permutation_to_soa(const prtl_perm_t& perm,
                                                 npart_t            n) {
    if (n == 0u) {
      return;
    }

    // The *_prev arrays (i{1,2,3}_prev, dx{1,2,3}_prev) are intentionally
    // NOT permuted. SortSpatially runs at the very end of the step loop
    // (engine step_forward), and the first thing the next step's pusher
    // does is overwrite prev := current (positionPush, sr.hpp / gr.hpp)
    // for every active particle, before any consumer reads prev:
    //   - current deposit: runs after the push, which has already
    //     overwritten prev; species with pusher==NONE (whose prev would
    //     stay un-permuted) are skipped by CurrentsDeposit entirely.
    //   - pusher getParticlePrevPosition / piston: read prev only after
    //     positionPush has rewritten it within the same call.
    //   - checkpoint (prev is checkpoint-only, never in diagnostic
    //     output): on restart the first push overwrites prev before it
    //     is read, so restart results are unaffected; only the redundant
    //     prev field saved to the checkpoint differs from the old code.
    // Permuting prev would therefore reorder data that is overwritten
    // before it is ever observed.
    //
    // Every permuted member has its own persistent, grow-only scratch
    // (`reserve_scratch_*` reallocates only when the live count outgrows the
    // capacity), so one kernel gathers all members and one copies them back,
    // queued asynchronously on the default (in-order) execution space; the
    // caller fences once at the end of the sort.
    const auto npr = static_cast<npart_t>(npld_r());
    const auto npi = static_cast<npart_t>(npld_i());
    for (auto d { 0u }; d < static_cast<unsigned>(D); ++d) {
      reserve_scratch_1d(m_sort_scratch_i[d], "perm_scratch_i", n);
      reserve_scratch_1d(m_sort_scratch_dx[d], "perm_scratch_dx", n);
    }
    const auto nu = (D == Dim::_2D and C != Coord::Cartesian) ? 5u : 4u;
    for (auto k { 0u }; k < nu; ++k) {
      reserve_scratch_1d(m_sort_scratch_u[k], "perm_scratch_u", n);
    }
    reserve_scratch_1d(m_sort_scratch_tag, "perm_scratch_tag", n);
    if (npr > 0u) {
      reserve_scratch_2d(m_sort_scratch_pld_r, "perm_scratch_pld_r", n, npr);
    }
    if (npi > 0u) {
      reserve_scratch_2d(m_sort_scratch_pld_i, "perm_scratch_pld_i", n, npi);
    }

    const auto fill = [&](auto& k) {
      k.a = *this;
      for (auto d { 0u }; d < 3u; ++d) {
        k.si[d]  = m_sort_scratch_i[d];
        k.sdx[d] = m_sort_scratch_dx[d];
      }
      for (auto u { 0u }; u < 5u; ++u) {
        k.su[u] = m_sort_scratch_u[u];
      }
      k.stag = m_sort_scratch_tag;
      k.spr  = m_sort_scratch_pld_r;
      k.spi  = m_sort_scratch_pld_i;
      k.perm = perm;
      k.npr  = npr;
      k.npi  = npi;
    };
    const auto policy =
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<npart_t>>(
        Kokkos::DefaultExecutionSpace {},
        0,
        n);
    permute_helpers::PermuteSoA_kernel<D, C, false> gather;
    fill(gather);
    Kokkos::parallel_for("PermuteGather", policy, gather);
    permute_helpers::PermuteSoA_kernel<D, C, true> back;
    fill(back);
    Kokkos::parallel_for("PermuteBack", policy, back);
  }
#endif // TILED_DEPOSIT_USE_VENDOR_SORT

#if defined(TILED_DEPOSIT_USE_VENDOR_SORT)
  #define APPLY_PERM_INSTANTIATE(D, C)                                          \
    template void Particles<D, C>::apply_permutation_to_soa(const prtl_perm_t&, \
                                                            npart_t);
#else
  #define APPLY_PERM_INSTANTIATE(D, C)
#endif

#define PARTICLES_SORT(D, C)                                                   \
  template void Particles<D, C>::RemoveDead();                                 \
  template void Particles<D, C>::SortSpatially(const Grid<D>&);                \
  APPLY_PERM_INSTANTIATE(D, C)

  PARTICLES_SORT(Dim::_1D, Coord::Cartesian)
  PARTICLES_SORT(Dim::_2D, Coord::Cartesian)
  PARTICLES_SORT(Dim::_3D, Coord::Cartesian)
  PARTICLES_SORT(Dim::_2D, Coord::Spherical)
  PARTICLES_SORT(Dim::_2D, Coord::Qspherical)
  PARTICLES_SORT(Dim::_3D, Coord::Spherical)
  PARTICLES_SORT(Dim::_3D, Coord::Qspherical)
#undef PARTICLES_SORT
#undef APPLY_PERM_INSTANTIATE

} // namespace ntt
