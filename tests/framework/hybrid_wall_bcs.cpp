/**
 * Reflecting-wall (x1 conductor) conditions of the hybrid engine on fields
 * filled with hashed values in every storage cell: FieldBoundaries (E and B
 * of em), WallEPrime (em0), WallScratchB (cur), WallBckpFill (bckp),
 * MomentsWallFill (em0 as a moments buffer) and MomentsWallBC with and
 * without the fold (aux). Domains with both walls (x1 not split) and with one
 * (x1 split), in 1D, 2D and 3D.
 *
 * Checks the mirror relations the wall conditions establish in the wall
 * ghosts of bckp and aux, and prints a hash of all fields, to compare builds
 * of different versions of the wall kernels bit by bit.
 */
#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"

#include "metrics/minkowski.h"

#include "engines/hybrid/fields_bcs.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

#include <Kokkos_Core.hpp>
#include <plog/Appenders/ConsoleAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Init.h>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ntt;

namespace {

  auto value(unsigned int idx, int f, const int i[3], int c) -> real_t {
    std::uint32_t h = 2166136261u;
    for (const auto v : { i[0], i[1], i[2], c, f, static_cast<int>(idx) }) {
      h = (h ^ static_cast<std::uint32_t>(v + 1000)) * 16777619u;
    }
    return static_cast<real_t>(static_cast<double>(h >> 8) / 16777216.0 - 0.5);
  }

  template <Dimension D, class V>
  auto at(const V& v, const int i[3], int c) -> decltype(auto) {
    if constexpr (D == Dim::_1D) {
      return v(i[0], c);
    } else if constexpr (D == Dim::_2D) {
      return v(i[0], i[1], c);
    } else {
      return v(i[0], i[1], i[2], c);
    }
  }

  template <Dimension D, class F>
  void for_cells(const std::vector<int>& e, F f) {
    for (int i1 = 0; i1 < e[0]; ++i1) {
      for (int i2 = 0; i2 < e[1]; ++i2) {
        for (int i3 = 0; i3 < e[2]; ++i3) {
          const int i[3] { i1, i2, i3 };
          f(i);
        }
      }
    }
  }

  template <Dimension D, unsigned short N>
  void fill(ndfield_t<D, N>& fld, unsigned int idx, int f, const std::vector<int>& e) {
    auto h = Kokkos::create_mirror_view(fld);
    for_cells<D>(e, [&](const int i[3]) {
      for (int c = 0; c < N; ++c) {
        at<D>(h, i, c) = value(idx, f, i, c);
      }
    });
    Kokkos::deep_copy(fld, h);
  }

  template <class V>
  void hash(std::uint64_t& h, const V& v) {
    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {}, v);
    const auto* b = reinterpret_cast<const unsigned char*>(m.data());
    for (std::size_t k = 0; k < m.span() * sizeof(typename V::value_type); ++k) {
      h = (h ^ b[k]) * 1099511628211ull;
    }
  }

  template <Dimension D>
  auto run(const std::vector<ncells_t>& res,
           const std::vector<int>&      decomposition,
           const std::string&           label) -> bool {
    using M  = metric::Minkowski<D>;
    int rank = 0, nranks = 1;
#if defined(MPI_ENABLED)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
#endif
    boundaries_t<real_t> extent;
    boundaries_t<FldsBC> fldsbc;
    boundaries_t<PrtlBC> prtlbc;
    for (auto d { 0u }; d < static_cast<unsigned>(D); ++d) {
      extent.push_back({ ZERO, static_cast<real_t>(res[d]) });
      if (d == 0u) {
        fldsbc.push_back({ FldsBC::CONDUCTOR, FldsBC::CONDUCTOR });
        prtlbc.push_back({ PrtlBC::REFLECT, PrtlBC::REFLECT });
      } else {
        fldsbc.push_back({ FldsBC::PERIODIC, FldsBC::PERIODIC });
        prtlbc.push_back({ PrtlBC::PERIODIC, PrtlBC::PERIODIC });
      }
    }
    Metadomain<SimEngine::HYBRID, M> metadomain { static_cast<unsigned int>(nranks),
                                                  decomposition,
                                                  res,
                                                  extent,
                                                  fldsbc,
                                                  prtlbc,
                                                  {},
                                                  {} };
    const auto       idx  = metadomain.l_subdomain_indices()[0];
    auto&            dom  = *metadomain.subdomain_ptr(idx);
    const auto&      grid = metadomain.mesh();
    std::vector<int> e { 1, 1, 1 };
    for (auto d { 0u }; d < static_cast<unsigned>(D); ++d) {
      e[d] = static_cast<int>(dom.mesh.n_active()[d] + 2 * N_GHOSTS);
    }
    fill<D, 6>(dom.fields.em, idx, 0, e);
    fill<D, 6>(dom.fields.em0, idx, 1, e);
    fill<D, 3>(dom.fields.cur, idx, 2, e);
    fill<D, 6>(dom.fields.bckp, idx, 3, e);
    fill<D, 6>(dom.fields.aux, idx, 4, e);

    hybrid::FieldBoundaries<D>(dom, grid, BC::E);
    hybrid::FieldBoundaries<D>(dom, grid, BC::B);
    hybrid::WallEPrime<D>(dom, grid);
    hybrid::WallScratchB<D>(dom, grid);
    hybrid::WallBckpFill<D>(dom, grid);
    hybrid::MomentsWallFill<D>(dom, grid, dom.fields.em0);
    hybrid::MomentsWallBC<D>(dom, grid, true);
    hybrid::MomentsWallBC<D>(dom, grid, false);

    // mirror relations in the wall ghosts: bckp E_x, B even and E_y, E_z
    // odd; aux V_x odd and V_y, V_z, N even
    long       nbad = 0;
    auto       bk   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {},
                                                  dom.fields.bckp);
    auto       ax   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {},
                                                  dom.fields.aux);
    const int  G    = static_cast<int>(N_GHOSTS);
    const auto off  = dom.offset_ncells();
    const bool lo   = (off[0] == 0u);
    const bool hi   = (off[0] + dom.mesh.n_active()[0] == grid.n_active()[0]);
    for_cells<D>(e, [&](const int i[3]) {
      int ia[3] { i[0], i[1], i[2] };
      if (lo and i[0] < G) {
        ia[0] = 2 * G - 1 - i[0];
      } else if (hi and i[0] >= e[0] - G) {
        ia[0] = 2 * (e[0] - G) - 1 - i[0];
      } else {
        return;
      }
      const real_t sb[6] { ONE, -ONE, -ONE, ONE, ONE, ONE };
      const real_t sa[4] { -ONE, ONE, ONE, ONE };
      for (int c = 0; c < 6; ++c) {
        nbad += (at<D>(bk, i, c) != sb[c] * at<D>(bk, ia, c)) ? 1 : 0;
      }
      for (int c = 0; c < 4; ++c) {
        nbad += (at<D>(ax, i, c) != sa[c] * at<D>(ax, ia, c)) ? 1 : 0;
      }
    });

    std::uint64_t h = 1469598103934665603ull;
    hash(h, dom.fields.em);
    hash(h, dom.fields.em0);
    hash(h, dom.fields.cur);
    hash(h, dom.fields.bckp);
    hash(h, dom.fields.aux);
#if defined(MPI_ENABLED)
    MPI_Allreduce(MPI_IN_PLACE, &nbad, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    std::vector<std::uint64_t> all(static_cast<std::size_t>(nranks));
    MPI_Gather(&h, 1, MPI_UINT64_T, all.data(), 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
#else
    std::vector<std::uint64_t> all { h };
#endif
    if (rank == 0) {
      std::cout << label << " (" << nranks << " ranks): mirror mismatches "
                << nbad << ", hash";
      for (const auto v : all) {
        std::cout << " " << std::hex << v << std::dec;
      }
      std::cout << ((nbad == 0) ? "  OK" : "  FAIL") << std::endl;
    }
    return nbad == 0;
  }

} // namespace

auto main(int argc, char* argv[]) -> int {
  GlobalInitialize(argc, argv);
  // errors raised inside the framework are reported through plog
  static plog::ConsoleAppender<plog::TxtFormatter> console;
  plog::init(plog::warning, &console);
  bool ok = true;
  try {
    ok &= run<Dim::_1D>({ 24 }, { -1 }, "1D x1 split");
    ok &= run<Dim::_2D>({ 16, 24 }, { 1, -1 }, "2D x2 split");
    ok &= run<Dim::_2D>({ 24, 16 }, { -1, 1 }, "2D x1 split");
    ok &= run<Dim::_3D>({ 12, 20, 20 }, { 1, -1, -1 }, "3D x2/x3 split");
    ok &= run<Dim::_3D>({ 24, 12, 12 }, { -1, 1, 1 }, "3D x1 split");
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    GlobalFinalize();
    return 1;
  }
  GlobalFinalize();
  return ok ? 0 : 1;
}
