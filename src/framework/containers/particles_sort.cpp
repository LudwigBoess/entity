#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/sorting.h"

#include "framework/containers/particles.h"
#include "framework/domain/grid.h"

#if defined(TEAM_POLICY)
  #include "utils/log.h"
  #include "utils/sort_dispatch.h"
#endif

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include <string>
#include <utility>
#include <vector>

namespace ntt {

  template <Dimension D, Coord::type C>
  auto Particles<D, C>::NpartsPerTagAndOffsets() const
    -> std::pair<std::vector<npart_t>, array_t<npart_t*>> {
    auto              this_tag = tag;
    const auto        num_tags = ntags();
    array_t<npart_t*> npptag { "nparts_per_tag", ntags() };

    // count # of particles per each tag
    auto npptag_scat = Kokkos::Experimental::create_scatter_view(npptag);
    Kokkos::parallel_for(
      "NpartPerTag",
      rangeActiveParticles(),
      Lambda(prtlidx_t p) {
        auto npptag_acc = npptag_scat.access();
        if (this_tag(p) < 0 || this_tag(p) >= static_cast<short>(num_tags)) {
          raise::KernelError(HERE, "Invalid tag value");
        }
        npptag_acc(this_tag(p)) += 1;
      });
    Kokkos::Experimental::contribute(npptag, npptag_scat);

    // copy the count to a vector on the host
    auto npptag_h = Kokkos::create_mirror_view(npptag);
    Kokkos::deep_copy(npptag_h, npptag);
    std::vector<npart_t> npptag_vec(num_tags);
    for (auto t { 0u }; t < num_tags; ++t) {
      npptag_vec[t] = npptag_h(t);
    }

    // count the offsets on the host and copy to device
    const array_t<npart_t*> tag_offsets("tag_offsets", num_tags - 3);
    auto tag_offsets_h = Kokkos::create_mirror_view(tag_offsets);

    tag_offsets_h(0) = npptag_vec[2]; // offset for tag = 3
    for (auto t { 1u }; t < num_tags - 3; ++t) {
      tag_offsets_h(t) = npptag_vec[t + 2] + tag_offsets_h(t - 1);
    }
    Kokkos::deep_copy(tag_offsets, tag_offsets_h);

    return { npptag_vec, tag_offsets };
  }

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

  template <Dimension D, Coord::type C>
  void Particles<D, C>::SortSpatially(const Grid<D>& grid) {
#if defined(TEAM_POLICY)
    // ---------------------- team_policy: tile-based sort ------------------ //
    // One-shot announcement (rank 0, once per process) of the active
    // sort backend — diagnostic for verifying that the compile-time
    // selection actually picked what was expected. Once-flag means this
    // costs essentially nothing on subsequent sort calls.
    static constexpr const char* k_sort_backend_name =
  #if defined(SYCL_ENABLED) && defined(ONEDPL_ENABLED)
      "OneDPL (SYCL)";
  #elif defined(CUDA_ENABLED) && defined(THRUST_ENABLED)
      "Thrust (CUDA)";
  #elif !defined(DEVICE_ENABLED)
      "StdSort (host)";
  #else
      "Kokkos::BinSort (fallback)";
  #endif
    info::Print(std::string("[team_policy] sort backend: ") +
                  k_sort_backend_name +
                  " (tile_size=" +
                  std::to_string(static_cast<int>(TEAM_POLICY_TILE_SIZE)) + ")",
                /*colored=*/false,
                /*stdout=*/true,
                /*once=*/true,
                /*info=*/true);

    const auto npart_local = npart();
    if (npart_local == 0u) {
      m_tile_layout = TileLayout<D> {};
      m_is_sorted   = true;
      return;
    }

    constexpr unsigned short T = static_cast<unsigned short>(
      TEAM_POLICY_TILE_SIZE);
    static_assert(T > 0u, "TEAM_POLICY_TILE_SIZE must be > 0");

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

    // 2. Compute per-particle tile key (with min(i, i_prev)) and per-tile
    //    counts in a single fused pass.
    array_t<ncells_t*> tile_indices { "tile_indices", npart_local };
    array_t<npart_t*>  num_ppt { "num_ppt", total_tiles };
    Kokkos::deep_copy(num_ppt, npart_t { 0u });

    Kokkos::parallel_for(
      "FillTileIndicesAndCount",
      rangeActiveParticles(),
      sort::PositionToTileIndex<D, true, true> { i1,
                                                  i2,
                                                  i3,
                                                  tag,
                                                  tile_indices,
                                                  ncells_active,
                                                  static_cast<ncells_t>(T),
                                                  num_ppt,
                                                  i1_prev,
                                                  i2_prev,
                                                  i3_prev });

    // 3. Prefix-sum num_ppt -> tile_offsets[0..total_tiles].
    array_t<npart_t*> tile_offsets { "tile_offsets", total_tiles + 1u };
    npart_t           alive_total { 0u };
    {
      auto num_ppt_v      = num_ppt;
      auto tile_offsets_v = tile_offsets;
      Kokkos::parallel_scan(
        "TileOffsetScan",
        static_cast<ncells_t>(total_tiles),
        KOKKOS_LAMBDA(const ncells_t t,
                      npart_t&       acc,
                      const bool     final_pass) {
          if (final_pass) {
            tile_offsets_v(t) = acc;
          }
          acc += num_ppt_v(t);
        },
        alive_total);
    }
    // Set tile_offsets[total_tiles] = alive_total.
    Kokkos::deep_copy(
      Kokkos::subview(tile_offsets,
                      std::make_pair(total_tiles, total_tiles + 1u)),
      alive_total);

    // 4. Build the permutation via the compile-time-selected backend.
    //    Sentinel bin for dead particles is total_tiles + 1u; reserve
    //    total_tiles + 2u bins for BinSort.
    const ncells_t n_bins = total_tiles + 2u;
    prtl_perm_t    perm { "tile_perm", npart_local };
  #if defined(SYCL_ENABLED) && defined(ONEDPL_ENABLED)
    sort_helpers::sort_by_key_dispatch(tile_indices,
                                       perm,
                                       n_bins,
                                       sort::backend::OneDPL {});
  #elif defined(CUDA_ENABLED) && defined(THRUST_ENABLED)
    sort_helpers::sort_by_key_dispatch(tile_indices,
                                       perm,
                                       n_bins,
                                       sort::backend::Thrust {});
  #elif !defined(DEVICE_ENABLED)
    sort_helpers::sort_by_key_dispatch(tile_indices,
                                       perm,
                                       n_bins,
                                       sort::backend::StdSort {});
  #else
    // Device build with no vendor sort library detected — use BinSort.
    sort_helpers::sort_by_key_dispatch(tile_indices,
                                       perm,
                                       n_bins,
                                       sort::backend::BinSort {});
  #endif

    // 5. Apply the permutation to all SoA arrays via a single fused gather
    //    + per-array deep_copy. After this, particle p's data lives at
    //    SoA index p (with p sorted by tile in [0, alive_total)).
    apply_permutation_to_soa(perm);

    // 6. Populate m_tile_layout for downstream consumers (tiled deposit /
    //    pusher kernels). tile_perm is retained for diagnostic purposes
    //    (after the gather above, the SoA arrays are already in tile
    //    order, so consumers iterate [tile_offsets(t), tile_offsets(t+1))
    //    directly without re-indirecting through tile_perm).
    m_tile_layout.ntiles_per_axis[0] = ntx[0];
    m_tile_layout.ntiles_per_axis[1] = ntx[1];
    m_tile_layout.ntiles_per_axis[2] = ntx[2];
    m_tile_layout.ntiles_total       = total_tiles;
    m_tile_layout.tile_size          = T;
    m_tile_layout.tile_offsets       = tile_offsets;
    m_tile_layout.tile_perm          = perm;
    m_is_sorted                      = true;
#else  // !TEAM_POLICY — legacy in-place BinSort by global cell index
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
    m_is_sorted = true;
#endif // TEAM_POLICY
  }

#if defined(TEAM_POLICY)
  template <Dimension D, Coord::type C>
  void Particles<D, C>::apply_permutation_to_soa(const prtl_perm_t& perm) {
    const auto n = npart();
    if (n == 0u) {
      return;
    }

    // Allocate scratch buffers for every SoA array. Total memory =
    // sizeof(particles)·n; transient — freed at end of this function.
    auto buf_i1       = array_t<int*>("buf_i1", n);
    auto buf_dx1      = array_t<prtldx_t*>("buf_dx1", n);
    auto buf_i1_prev  = array_t<int*>("buf_i1_prev", n);
    auto buf_dx1_prev = array_t<prtldx_t*>("buf_dx1_prev", n);
    array_t<int*>      buf_i2, buf_i2_prev, buf_i3, buf_i3_prev;
    array_t<prtldx_t*> buf_dx2, buf_dx2_prev, buf_dx3, buf_dx3_prev;
    if constexpr (D == Dim::_2D or D == Dim::_3D) {
      buf_i2       = array_t<int*>("buf_i2", n);
      buf_dx2      = array_t<prtldx_t*>("buf_dx2", n);
      buf_i2_prev  = array_t<int*>("buf_i2_prev", n);
      buf_dx2_prev = array_t<prtldx_t*>("buf_dx2_prev", n);
    }
    if constexpr (D == Dim::_3D) {
      buf_i3       = array_t<int*>("buf_i3", n);
      buf_dx3      = array_t<prtldx_t*>("buf_dx3", n);
      buf_i3_prev  = array_t<int*>("buf_i3_prev", n);
      buf_dx3_prev = array_t<prtldx_t*>("buf_dx3_prev", n);
    }
    auto buf_ux1    = array_t<real_t*>("buf_ux1", n);
    auto buf_ux2    = array_t<real_t*>("buf_ux2", n);
    auto buf_ux3    = array_t<real_t*>("buf_ux3", n);
    auto buf_weight = array_t<real_t*>("buf_weight", n);
    auto buf_tag    = array_t<short*>("buf_tag", n);
    array_t<real_t*> buf_phi;
    if constexpr (D == Dim::_2D and C != Coord::Cartesian) {
      buf_phi = array_t<real_t*>("buf_phi", n);
    }
    const auto         nplr = npld_r();
    const auto         npli = npld_i();
    array_t<real_t**>  buf_pld_r;
    array_t<npart_t**> buf_pld_i;
    if (nplr > 0) {
      buf_pld_r = array_t<real_t**>("buf_pld_r", n, nplr);
    }
    if (npli > 0) {
      buf_pld_i = array_t<npart_t**>("buf_pld_i", n, npli);
    }

    // Local references for capture-by-value into the device lambda.
    auto& s_i1     = i1;
    auto& s_dx1    = dx1;
    auto& s_i1p    = i1_prev;
    auto& s_dx1p   = dx1_prev;
    auto& s_i2     = i2;
    auto& s_dx2    = dx2;
    auto& s_i2p    = i2_prev;
    auto& s_dx2p   = dx2_prev;
    auto& s_i3     = i3;
    auto& s_dx3    = dx3;
    auto& s_i3p    = i3_prev;
    auto& s_dx3p   = dx3_prev;
    auto& s_ux1    = ux1;
    auto& s_ux2    = ux2;
    auto& s_ux3    = ux3;
    auto& s_weight = weight;
    auto& s_tag    = tag;
    auto& s_phi    = phi;
    auto& s_pld_r  = pld_r;
    auto& s_pld_i  = pld_i;
    auto& s_perm   = perm;

    // Single fused gather over all 1D-shaped SoA members.
    Kokkos::parallel_for(
      "GatherByPerm",
      rangeActiveParticles(),
      Lambda(prtlidx_t p) {
        const auto src = s_perm(p);
        buf_i1(p)      = s_i1(src);
        buf_dx1(p)     = s_dx1(src);
        buf_i1_prev(p) = s_i1p(src);
        buf_dx1_prev(p) = s_dx1p(src);
        if constexpr (D == Dim::_2D or D == Dim::_3D) {
          buf_i2(p)       = s_i2(src);
          buf_dx2(p)      = s_dx2(src);
          buf_i2_prev(p)  = s_i2p(src);
          buf_dx2_prev(p) = s_dx2p(src);
        }
        if constexpr (D == Dim::_3D) {
          buf_i3(p)       = s_i3(src);
          buf_dx3(p)      = s_dx3(src);
          buf_i3_prev(p)  = s_i3p(src);
          buf_dx3_prev(p) = s_dx3p(src);
        }
        buf_ux1(p)    = s_ux1(src);
        buf_ux2(p)    = s_ux2(src);
        buf_ux3(p)    = s_ux3(src);
        buf_weight(p) = s_weight(src);
        buf_tag(p)    = s_tag(src);
        if constexpr (D == Dim::_2D and C != Coord::Cartesian) {
          buf_phi(p) = s_phi(src);
        }
      });

    // 2D payload arrays — separate kernels.
    if (nplr > 0) {
      auto       buf_local  = buf_pld_r;
      const auto perm_local = perm;
      auto&      src_local  = pld_r;
      Kokkos::parallel_for(
        "GatherByPermPldR",
        CreateParticleRangePolicy<Dim::_2D>({ 0u, 0u },
                                            { n, static_cast<npart_t>(nplr) }),
        Lambda(prtlidx_t p, npart_t l) {
          buf_local(p, l) = src_local(perm_local(p), l);
        });
    }
    if (npli > 0) {
      auto       buf_local  = buf_pld_i;
      const auto perm_local = perm;
      auto&      src_local  = pld_i;
      Kokkos::parallel_for(
        "GatherByPermPldI",
        CreateParticleRangePolicy<Dim::_2D>({ 0u, 0u },
                                            { n, static_cast<npart_t>(npli) }),
        Lambda(prtlidx_t p, npart_t l) {
          buf_local(p, l) = src_local(perm_local(p), l);
        });
    }

    // Deep-copy buffers back into member arrays (touches only [0, n)).
    const auto slice = prtl_slice_t(0u, n);
    Kokkos::deep_copy(Kokkos::subview(i1, slice), buf_i1);
    Kokkos::deep_copy(Kokkos::subview(dx1, slice), buf_dx1);
    Kokkos::deep_copy(Kokkos::subview(i1_prev, slice), buf_i1_prev);
    Kokkos::deep_copy(Kokkos::subview(dx1_prev, slice), buf_dx1_prev);
    if constexpr (D == Dim::_2D or D == Dim::_3D) {
      Kokkos::deep_copy(Kokkos::subview(i2, slice), buf_i2);
      Kokkos::deep_copy(Kokkos::subview(dx2, slice), buf_dx2);
      Kokkos::deep_copy(Kokkos::subview(i2_prev, slice), buf_i2_prev);
      Kokkos::deep_copy(Kokkos::subview(dx2_prev, slice), buf_dx2_prev);
    }
    if constexpr (D == Dim::_3D) {
      Kokkos::deep_copy(Kokkos::subview(i3, slice), buf_i3);
      Kokkos::deep_copy(Kokkos::subview(dx3, slice), buf_dx3);
      Kokkos::deep_copy(Kokkos::subview(i3_prev, slice), buf_i3_prev);
      Kokkos::deep_copy(Kokkos::subview(dx3_prev, slice), buf_dx3_prev);
    }
    Kokkos::deep_copy(Kokkos::subview(ux1, slice), buf_ux1);
    Kokkos::deep_copy(Kokkos::subview(ux2, slice), buf_ux2);
    Kokkos::deep_copy(Kokkos::subview(ux3, slice), buf_ux3);
    Kokkos::deep_copy(Kokkos::subview(weight, slice), buf_weight);
    Kokkos::deep_copy(Kokkos::subview(tag, slice), buf_tag);
    if constexpr (D == Dim::_2D and C != Coord::Cartesian) {
      Kokkos::deep_copy(Kokkos::subview(phi, slice), buf_phi);
    }
    if (nplr > 0) {
      Kokkos::deep_copy(Kokkos::subview(pld_r, slice, Kokkos::ALL), buf_pld_r);
    }
    if (npli > 0) {
      Kokkos::deep_copy(Kokkos::subview(pld_i, slice, Kokkos::ALL), buf_pld_i);
    }
  }
#endif // TEAM_POLICY

#if defined(TEAM_POLICY)
  #define TEAM_POLICY_INSTANTIATE_APPLY(D, C)                                    \
    template void Particles<D, C>::apply_permutation_to_soa(                   \
      const prtl_perm_t&);
#else
  #define TEAM_POLICY_INSTANTIATE_APPLY(D, C)
#endif

#define PARTICLES_SORT(D, C)                                                   \
  template auto Particles<D, C>::NpartsPerTagAndOffsets() const                \
    -> std::pair<std::vector<npart_t>, array_t<npart_t*>>;                     \
  template void Particles<D, C>::RemoveDead();                                 \
  template void Particles<D, C>::SortSpatially(const Grid<D>&);                \
  TEAM_POLICY_INSTANTIATE_APPLY(D, C)

  PARTICLES_SORT(Dim::_1D, Coord::Cartesian)
  PARTICLES_SORT(Dim::_2D, Coord::Cartesian)
  PARTICLES_SORT(Dim::_3D, Coord::Cartesian)
  PARTICLES_SORT(Dim::_2D, Coord::Spherical)
  PARTICLES_SORT(Dim::_2D, Coord::Qspherical)
  PARTICLES_SORT(Dim::_3D, Coord::Spherical)
  PARTICLES_SORT(Dim::_3D, Coord::Qspherical)
#undef PARTICLES_SORT
#undef TEAM_POLICY_INSTANTIATE_APPLY

} // namespace ntt
