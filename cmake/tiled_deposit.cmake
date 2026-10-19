list(FIND tiled_deposit_tile_sizes "${tiled_deposit_tile_size}" _tps_idx)
if(_tps_idx EQUAL -1)
  message(
    FATAL_ERROR
      "${Red}tiled_deposit_tile_size must be one of ${tiled_deposit_tile_sizes}, "
      "got '${tiled_deposit_tile_size}'${ColorReset}")
endif()
add_compile_options("-D TILED_DEPOSIT")
add_compile_options("-D TILED_DEPOSIT_TILE_SIZE=${tiled_deposit_tile_size}")

# Tile edge length of the 3D runs, if different (0: tiled_deposit_tile_size).
# The 3D tile scratch holds (T + 2 HALO)^3 cells, so a smaller 3D tile lets
# more teams share the shared local memory of a compute unit.
if(NOT "${tiled_deposit_tile_size_3d}" STREQUAL "0")
  list(FIND tiled_deposit_tile_sizes "${tiled_deposit_tile_size_3d}" _tps3_idx)
  if(_tps3_idx EQUAL -1)
    message(
      FATAL_ERROR
        "${Red}tiled_deposit_tile_size_3d must be 0 or one of "
        "${tiled_deposit_tile_sizes}, got '${tiled_deposit_tile_size_3d}'${ColorReset}")
  endif()
  add_compile_options("-D TILED_DEPOSIT_TILE_SIZE_3D=${tiled_deposit_tile_size_3d}")
endif()

# Compile-time tiled-deposit scratch halo drift. Sizes the halo so a particle
# that drifts up to DRIFT cells between two sorts still deposits inside its tile
# scratch; particles drifting further take the per-particle global-J escape
# valve (correct, only slower). This is independent of the sort cadence, which
# is set at runtime via `spatial_sorting_interval`. Defaults to 1 (the
# sorted-every-step case).
add_compile_options("-D TILED_DEPOSIT_DRIFT=${tiled_deposit_drift}")

# Accumulate the tile scratch of the hybrid moment deposit in two-word fixed
# point with native 32-bit integer atomics instead of real_t atomic adds (on
# Intel PVC a float atomic add on SLM is a compare-and-swap retry loop). Changes
# the rounding of the deposited moments (see kernels/tiled_scatter.hpp).
if(${tiled_deposit_fixed_point})
  add_compile_options("-D TILED_DEPOSIT_FIXED_POINT")
endif()
