#include "enums.h"
#include "global.h"

#include "utils/error.h"

#include "framework/containers/particles.h"
#include "framework/domain/grid.h"

#include <iostream>
#include <utility>
#include <vector>

auto main(int argc, char* argv[]) -> int {
  ntt::GlobalInitialize(argc, argv);
  try {
    {
      // 2D
      auto grid = ntt::Grid<Dim::_2D> {
        {           10u,             30u },
        { { -5.0, 5.0 }, { -15.0, 15.0 } }
      };
      auto prtls = ntt::Particles<Dim::_2D, ntt::Coord::Cartesian>(
        1,
        "test",
        1.0f,
        1.0f,
        100u,
        0u,
        0u,
        ntt::ParticlePusher::BORIS,
        false,
        ntt::RadiativeDrag::NONE,
        ntt::EmissionType::NONE,
        2u,
        1u);
      auto& i1_p      = prtls.i1;
      auto& i2_p      = prtls.i2;
      auto& i1_prev_p = prtls.i1_prev;
      auto& i2_prev_p = prtls.i2_prev;
      auto& tag_p     = prtls.tag;
      auto& weight_p  = prtls.weight;
      auto& pld_r     = prtls.pld_r;
      auto& pld_i     = prtls.pld_i;
      auto& dx1_p     = prtls.dx1;
      auto& dx2_p     = prtls.dx2;
      auto& ux1_p     = prtls.ux1;
      auto& ux2_p     = prtls.ux2;
      auto& ux3_p     = prtls.ux3;
      Kokkos::parallel_for(
        "InitParticles",
        prtls.maxnpart(),
        Lambda(prtlidx_t p) {
          if (p < 66u) {
            tag_p(p) = (p % 10u == 0u) ? ntt::ParticleTag::dead
                                       : ntt::ParticleTag::alive;
            if (p % 4u == 0u) {
              i1_p(p)     = 8u;
              i2_p(p)     = 2u;
              weight_p(p) = 0.0;
            } else if (p % 4u == 1u) {
              i1_p(p)     = 2u;
              i2_p(p)     = 8u;
              weight_p(p) = 1.0;
            } else if (p % 4u == 2u) {
              i1_p(p)     = 5u;
              i2_p(p)     = 15u;
              weight_p(p) = 2.0;
            } else {
              i1_p(p)     = 0u;
              i2_p(p)     = 23u;
              weight_p(p) = 3.0;
            }
            // tiled_deposit keys on min(i, i_prev); without a meaningful
            // i_prev every key would collapse to 0. Set i_prev = i so the
            // tile key reduces to the particle's current cell.
            i1_prev_p(p) = i1_p(p);
            i2_prev_p(p) = i2_p(p);
            pld_r(p, 0)  = weight_p(p) + static_cast<real_t>(0.5);
            pld_r(p, 1)  = weight_p(p) + static_cast<real_t>(10.5);
            pld_i(p, 0)  = static_cast<npart_t>(weight_p(p) + 10.0);
            // every other member a function of the weight, to check that all
            // are permuted alike
            dx1_p(p)     = static_cast<prtldx_t>(0.125 * weight_p(p));
            dx2_p(p)     = static_cast<prtldx_t>(0.0625 * weight_p(p));
            ux1_p(p)     = weight_p(p) + static_cast<real_t>(20.0);
            ux2_p(p)     = weight_p(p) + static_cast<real_t>(30.0);
            ux3_p(p)     = weight_p(p) + static_cast<real_t>(40.0);
          } else {
            tag_p(p) = ntt::ParticleTag::dead;
          }
          if (tag_p(p) == ntt::ParticleTag::dead) {
            weight_p(p) = -1.0;
          }
        });
      prtls.set_npart(66u);

      prtls.SortSpatially(grid);

      auto i1_h     = Kokkos::create_mirror_view(prtls.i1);
      auto i2_h     = Kokkos::create_mirror_view(prtls.i2);
      auto tag_h    = Kokkos::create_mirror_view(prtls.tag);
      auto weight_h = Kokkos::create_mirror_view(prtls.weight);
      auto pld_r_h  = Kokkos::create_mirror_view(prtls.pld_r);
      auto pld_i_h  = Kokkos::create_mirror_view(prtls.pld_i);
      auto dx1_h    = Kokkos::create_mirror_view(prtls.dx1);
      auto dx2_h    = Kokkos::create_mirror_view(prtls.dx2);
      auto ux1_h    = Kokkos::create_mirror_view(prtls.ux1);
      auto ux2_h    = Kokkos::create_mirror_view(prtls.ux2);
      auto ux3_h    = Kokkos::create_mirror_view(prtls.ux3);
      Kokkos::deep_copy(i1_h, prtls.i1);
      Kokkos::deep_copy(i2_h, prtls.i2);
      Kokkos::deep_copy(tag_h, prtls.tag);
      Kokkos::deep_copy(weight_h, prtls.weight);
      Kokkos::deep_copy(pld_r_h, prtls.pld_r);
      Kokkos::deep_copy(pld_i_h, prtls.pld_i);
      Kokkos::deep_copy(dx1_h, prtls.dx1);
      Kokkos::deep_copy(dx2_h, prtls.dx2);
      Kokkos::deep_copy(ux1_h, prtls.ux1);
      Kokkos::deep_copy(ux2_h, prtls.ux2);
      Kokkos::deep_copy(ux3_h, prtls.ux3);

      // Tile geometry, mirroring sort::PositionToTileIndex. T = 1 (no
      // tiled_deposit) reproduces the legacy per-cell ordering.
#if defined(TILED_DEPOSIT)
      const ncells_t T = static_cast<ncells_t>(TILED_DEPOSIT_TILE_SIZE);
#else
      const ncells_t T = 1u;
#endif
      const auto     na      = grid.n_active();
      const ncells_t ntx2    = (na[1] + T - 1u) / T;
      const auto     tile_of = [&](int a, int b) -> ncells_t {
        return (static_cast<ncells_t>(a) / T) * ntx2 +
               (static_cast<ncells_t>(b) / T);
      };

      // SortSpatially is order-by-tile, not order-by-cell: assert the
      // invariants that hold for any tile size rather than a hardwired
      // permutation. (1) alive particles form a prefix sorted by
      // non-decreasing tile index; (2) every SoA member is permuted by the
      // *same* permutation, so each alive slot still satisfies
      // pld == f(weight); (3) no alive particle is lost. Only [0, npart())
      // is defined after a sort. The tiled_deposit path compacts — it drops
      // the dead, so npart() equals the alive count and [0, npart()) is
      // entirely alive; the legacy (non-team) path keeps the dead as a
      // weight == -1 suffix, leaving npart() unchanged. Iterating
      // [0, npart()) exercises both: the prefix-sorted / no-alive-after-dead
      // checks below hold either way.
#if defined(TILED_DEPOSIT)
      raise::ErrorIf(prtls.npart() != 59u,
                     "tiled_deposit sort must compact: npart() should equal "
                     "the alive count",
                     HERE);
#else
      raise::ErrorIf(prtls.npart() != 66u,
                     "legacy sort should leave npart() unchanged",
                     HERE);
#endif
      bool     seen_dead   = false;
      bool     have_prev   = false;
      ncells_t prev_tile   = 0u;
      npart_t  n_alive_obs = 0u;
      for (auto p { 0u }; p < prtls.npart(); ++p) {
        if (tag_h(p) != ntt::ParticleTag::alive) {
          seen_dead = true;
          raise::ErrorIf(weight_h(p) != -1.0,
                         "dead particle has unexpected weight",
                         HERE);
          continue;
        }
        raise::ErrorIf(seen_dead,
                       "alive particle after a dead one (not sorted to prefix)",
                       HERE);
        const auto tile = tile_of(i1_h(p), i2_h(p));
        raise::ErrorIf(have_prev && (tile < prev_tile),
                       "alive particles not sorted by tile index",
                       HERE);
        prev_tile = tile;
        have_prev = true;
        ++n_alive_obs;
        raise::ErrorIf(pld_r_h(p, 0) != weight_h(p) + static_cast<real_t>(0.5),
                       "error in sorting particle real payload 0",
                       HERE);
        raise::ErrorIf(pld_r_h(p, 1) != weight_h(p) + static_cast<real_t>(10.5),
                       "error in sorting particle real payload 1",
                       HERE);
        raise::ErrorIf(pld_i_h(p, 0) != static_cast<npart_t>(
                                          weight_h(p) + static_cast<real_t>(10.0)),
                       "error in sorting particle integer payload 0",
                       HERE);
        raise::ErrorIf(dx1_h(p) != static_cast<prtldx_t>(0.125 * weight_h(p)) or
                         dx2_h(p) != static_cast<prtldx_t>(0.0625 * weight_h(p)),
                       "error in sorting particle displacements",
                       HERE);
        raise::ErrorIf(ux1_h(p) != weight_h(p) + static_cast<real_t>(20.0) or
                         ux2_h(p) != weight_h(p) + static_cast<real_t>(30.0) or
                         ux3_h(p) != weight_h(p) + static_cast<real_t>(40.0),
                       "error in sorting particle velocities",
                       HERE);
      }
      raise::ErrorIf(n_alive_obs != 59u,
                     "wrong number of alive particles after sort",
                     HERE);
    }
    {
      // 3D
      auto grid = ntt::Grid<Dim::_3D> {
        {            6u,            7u,            8u },
        { { -3.0, 3.0 }, { -3.5, 3.5 }, { -4.0, 4.0 } }
      };
      auto prtls = ntt::Particles<Dim::_3D, ntt::Coord::Cartesian>(
        1,
        "test",
        1.0f,
        1.0f,
        100u,
        0u,
        0u,
        ntt::ParticlePusher::BORIS,
        false,
        ntt::RadiativeDrag::NONE,
        ntt::EmissionType::NONE,
        0u,
        0u);
      auto& i1_p      = prtls.i1;
      auto& i2_p      = prtls.i2;
      auto& i3_p      = prtls.i3;
      auto& i1_prev_p = prtls.i1_prev;
      auto& i2_prev_p = prtls.i2_prev;
      auto& i3_prev_p = prtls.i3_prev;
      auto& tag_p     = prtls.tag;
      auto& weight_p  = prtls.weight;
      auto& dx1_p     = prtls.dx1;
      auto& dx2_p     = prtls.dx2;
      auto& dx3_p     = prtls.dx3;
      auto& ux1_p     = prtls.ux1;
      auto& ux2_p     = prtls.ux2;
      auto& ux3_p     = prtls.ux3;
      Kokkos::parallel_for(
        "InitParticles",
        prtls.maxnpart(),
        Lambda(prtlidx_t p) {
          if (p < 66u) {
            tag_p(p) = (p % 10u == 0u) ? ntt::ParticleTag::dead
                                       : ntt::ParticleTag::alive;
            if (p % 5u == 0u) {
              i1_p(p)     = 3u;
              i2_p(p)     = 2u;
              i3_p(p)     = 7u;
              weight_p(p) = 0.0;
            } else if (p % 5u == 1u) {
              i1_p(p)     = 2u;
              i2_p(p)     = 4u;
              i3_p(p)     = 3u;
              weight_p(p) = 1.0;
            } else if (p % 5u == 2u) {
              i1_p(p)     = 2u;
              i2_p(p)     = 6u;
              i3_p(p)     = 6u;
              weight_p(p) = 2.0;
            } else if (p % 5u == 3u) {
              i1_p(p)     = 3u;
              i2_p(p)     = 6u;
              i3_p(p)     = 6u;
              weight_p(p) = 3.0;
            } else {
              i1_p(p)     = 0u;
              i2_p(p)     = 6u;
              i3_p(p)     = 7u;
              weight_p(p) = 4.0;
            }
            // see 2D block: i_prev = i so the tiled_deposit tile key reduces
            // to the particle's current cell.
            i1_prev_p(p) = i1_p(p);
            i2_prev_p(p) = i2_p(p);
            i3_prev_p(p) = i3_p(p);
            dx1_p(p)     = static_cast<prtldx_t>(0.125 * weight_p(p));
            dx2_p(p)     = static_cast<prtldx_t>(0.0625 * weight_p(p));
            dx3_p(p)     = static_cast<prtldx_t>(0.03125 * weight_p(p));
            ux1_p(p)     = weight_p(p) + static_cast<real_t>(20.0);
            ux2_p(p)     = weight_p(p) + static_cast<real_t>(30.0);
            ux3_p(p)     = weight_p(p) + static_cast<real_t>(40.0);
          } else {
            tag_p(p) = ntt::ParticleTag::dead;
          }
          if (tag_p(p) == ntt::ParticleTag::dead) {
            weight_p(p) = -1.0;
          }
        });
      prtls.set_npart(66u);

      prtls.SortSpatially(grid);

      auto i1_h     = Kokkos::create_mirror_view(prtls.i1);
      auto i2_h     = Kokkos::create_mirror_view(prtls.i2);
      auto i3_h     = Kokkos::create_mirror_view(prtls.i3);
      auto tag_h    = Kokkos::create_mirror_view(prtls.tag);
      auto weight_h = Kokkos::create_mirror_view(prtls.weight);
      auto dx1_h    = Kokkos::create_mirror_view(prtls.dx1);
      auto dx2_h    = Kokkos::create_mirror_view(prtls.dx2);
      auto dx3_h    = Kokkos::create_mirror_view(prtls.dx3);
      auto ux1_h    = Kokkos::create_mirror_view(prtls.ux1);
      auto ux2_h    = Kokkos::create_mirror_view(prtls.ux2);
      auto ux3_h    = Kokkos::create_mirror_view(prtls.ux3);
      Kokkos::deep_copy(i1_h, prtls.i1);
      Kokkos::deep_copy(i2_h, prtls.i2);
      Kokkos::deep_copy(i3_h, prtls.i3);
      Kokkos::deep_copy(tag_h, prtls.tag);
      Kokkos::deep_copy(weight_h, prtls.weight);
      Kokkos::deep_copy(dx1_h, prtls.dx1);
      Kokkos::deep_copy(dx2_h, prtls.dx2);
      Kokkos::deep_copy(dx3_h, prtls.dx3);
      Kokkos::deep_copy(ux1_h, prtls.ux1);
      Kokkos::deep_copy(ux2_h, prtls.ux2);
      Kokkos::deep_copy(ux3_h, prtls.ux3);

      // Same invariants as the 2D block (no payloads here): alive prefix
      // sorted by non-decreasing tile index, alive count preserved. The
      // tiled_deposit path compacts the dead away (npart() == alive count);
      // the legacy path keeps them as a weight == -1 suffix. T = 1
      // reproduces the legacy per-cell order.
#if defined(TILED_DEPOSIT)
      const ncells_t T = static_cast<ncells_t>(TILED_DEPOSIT_TILE_SIZE);
#else
      const ncells_t T = 1u;
#endif
      const auto     na      = grid.n_active();
      const ncells_t ntx2    = (na[1] + T - 1u) / T;
      const ncells_t ntx3    = (na[2] + T - 1u) / T;
      const auto     tile_of = [&](int a, int b, int c) -> ncells_t {
        return ((static_cast<ncells_t>(a) / T) * ntx2 +
                (static_cast<ncells_t>(b) / T)) *
                 ntx3 +
               (static_cast<ncells_t>(c) / T);
      };

#if defined(TILED_DEPOSIT)
      raise::ErrorIf(prtls.npart() != 59u,
                     "tiled_deposit sort must compact: npart() should equal "
                     "the alive count",
                     HERE);
#else
      raise::ErrorIf(prtls.npart() != 66u,
                     "legacy sort should leave npart() unchanged",
                     HERE);
#endif
      bool     seen_dead   = false;
      bool     have_prev   = false;
      ncells_t prev_tile   = 0u;
      npart_t  n_alive_obs = 0u;
      for (auto p { 0u }; p < prtls.npart(); ++p) {
        if (tag_h(p) != ntt::ParticleTag::alive) {
          seen_dead = true;
          raise::ErrorIf(weight_h(p) != -1.0,
                         "dead particle has unexpected weight",
                         HERE);
          continue;
        }
        raise::ErrorIf(seen_dead,
                       "alive particle after a dead one (not sorted to prefix)",
                       HERE);
        const auto tile = tile_of(i1_h(p), i2_h(p), i3_h(p));
        raise::ErrorIf(have_prev && (tile < prev_tile),
                       "alive particles not sorted by tile index",
                       HERE);
        prev_tile = tile;
        have_prev = true;
        ++n_alive_obs;
        raise::ErrorIf(dx1_h(p) != static_cast<prtldx_t>(0.125 * weight_h(p)) or
                         dx2_h(p) != static_cast<prtldx_t>(0.0625 * weight_h(p)) or
                         dx3_h(p) != static_cast<prtldx_t>(0.03125 * weight_h(p)),
                       "error in sorting particle displacements",
                       HERE);
        raise::ErrorIf(ux1_h(p) != weight_h(p) + static_cast<real_t>(20.0) or
                         ux2_h(p) != weight_h(p) + static_cast<real_t>(30.0) or
                         ux3_h(p) != weight_h(p) + static_cast<real_t>(40.0),
                       "error in sorting particle velocities",
                       HERE);
      }
      raise::ErrorIf(n_alive_obs != 59u,
                     "wrong number of alive particles after sort",
                     HERE);
    }
#if defined(TILED_DEPOSIT)
    // tile layout: nt1 x nt2 tiles of which only some hold particles; the
    // tiles own consecutive slices of the sorted particles (12 and 300
    // tiles: the sort keys fit in 8 and in 16 bits)
    for (const auto nt : {
           std::pair<ncells_t, ncells_t> {  4u,  3u },
           std::pair<ncells_t, ncells_t> { 20u, 15u }
    }) {
      const ncells_t T      = static_cast<ncells_t>(TILED_DEPOSIT_TILE_SIZE);
      const ncells_t nt1    = nt.first;
      const ncells_t nt2    = nt.second;
      const ncells_t ntiles = nt1 * nt2;
      auto           grid   = ntt::Grid<Dim::_2D> {
        {                               nt1 * T,nt2 * T                                      },
        { { 0.0, static_cast<real_t>(nt1 * T) },
         { 0.0, static_cast<real_t>(nt2 * T) } }
      };
      const npart_t n0    = 5u * static_cast<npart_t>(ntiles);
      auto          prtls = ntt::Particles<Dim::_2D, ntt::Coord::Cartesian>(
        1,
        "test",
        1.0f,
        1.0f,
        n0 + 10u,
        0u,
        0u,
        ntt::ParticlePusher::BORIS,
        false,
        ntt::RadiativeDrag::NONE,
        ntt::EmissionType::NONE,
        0u,
        0u);
      auto                 i1_h  = Kokkos::create_mirror_view(prtls.i1);
      auto                 i2_h  = Kokkos::create_mirror_view(prtls.i2);
      auto                 i1p_h = Kokkos::create_mirror_view(prtls.i1_prev);
      auto                 i2p_h = Kokkos::create_mirror_view(prtls.i2_prev);
      auto                 tag_h = Kokkos::create_mirror_view(prtls.tag);
      auto                 w_h   = Kokkos::create_mirror_view(prtls.weight);
      std::vector<npart_t> count(ntiles, 0u);
      for (npart_t p = 0; p < n0; ++p) {
        // every third tile stays empty
        auto t = static_cast<ncells_t>((p * 7919u) % ntiles);
        if (t % 3u == 1u) {
          t = (t + 1u) % ntiles;
        }
        const auto tx = t / nt2;
        const auto ty = t % nt2;
        i1_h(p)       = static_cast<int>(tx * T + (p * 3u) % T);
        i2_h(p)       = static_cast<int>(ty * T + (p * 5u) % T);
        i1p_h(p)      = i1_h(p);
        i2p_h(p)      = i2_h(p);
        tag_h(p)      = (p % 7u == 3u) ? ntt::ParticleTag::dead
                                       : ntt::ParticleTag::alive;
        w_h(p)        = static_cast<real_t>(p);
        if (tag_h(p) == ntt::ParticleTag::alive) {
          ++count[static_cast<std::size_t>(t)];
        }
      }
      Kokkos::deep_copy(prtls.i1, i1_h);
      Kokkos::deep_copy(prtls.i2, i2_h);
      Kokkos::deep_copy(prtls.i1_prev, i1p_h);
      Kokkos::deep_copy(prtls.i2_prev, i2p_h);
      Kokkos::deep_copy(prtls.tag, tag_h);
      Kokkos::deep_copy(prtls.weight, w_h);
      prtls.set_npart(n0);
      prtls.SortSpatially(grid);

      const auto& layout = prtls.tile_layout();
      raise::ErrorIf(layout.ntiles_total != ntiles, "wrong number of tiles", HERE);
      auto off_h = Kokkos::create_mirror_view(layout.tile_offsets);
      Kokkos::deep_copy(off_h, layout.tile_offsets);
      Kokkos::deep_copy(i1_h, prtls.i1);
      Kokkos::deep_copy(i2_h, prtls.i2);
      Kokkos::deep_copy(w_h, prtls.weight);
      npart_t n_alive = 0u;
      for (const auto c : count) {
        n_alive += c;
      }
      raise::ErrorIf(prtls.npart() != n_alive or layout.npart_partitioned != n_alive,
                     "tile layout: npart / npart_partitioned != alive count",
                     HERE);
      raise::ErrorIf(off_h(0) != 0u or off_h(ntiles) != n_alive,
                     "tile layout: offsets do not span the alive particles",
                     HERE);
      for (auto t { 0u }; t < ntiles; ++t) {
        raise::ErrorIf(off_h(t + 1) - off_h(t) != count[t],
                       "tile layout: wrong particle count of a tile",
                       HERE);
        real_t w_prev = -ONE;
        for (auto p = off_h(t); p < off_h(t + 1); ++p) {
          const auto tile = (static_cast<ncells_t>(i1_h(p)) / T) * nt2 +
                            static_cast<ncells_t>(i2_h(p)) / T;
          raise::ErrorIf(tile != t, "tile layout: particle in the wrong tile", HERE);
          // stable sort: within a tile the original order is kept
          raise::ErrorIf(w_h(p) <= w_prev,
                         "tile layout: order within a tile not kept",
                         HERE);
          w_prev = w_h(p);
        }
      }
    }
#endif

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    ntt::GlobalFinalize();
    return 1;
  }
  ntt::GlobalFinalize();
  return 0;
}
