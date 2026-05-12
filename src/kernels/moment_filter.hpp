/**
 * @file kernels/moment_filter.hpp
 * @brief Generic binomial smoother for selected slots of an ndfield_t<D, N>
 * @implements
 *   - kernel::MomentFilter_kernel<>
 * @namespaces:
 *   - kernel::
 * @note
 *   Tensor-product binomial filter on each requested slot, separate for
 *   1D (3-pt), 2D (9-pt) and 3D (27-pt). Cartesian only. The caller is
 *   responsible for: (1) holding a snapshot of the input in `buffer`,
 *   (2) writing the filtered result into `array`, and (3) synchronizing
 *   the halo on `array` after each pass.
 */

#ifndef KERNELS_MOMENT_FILTER_HPP
#define KERNELS_MOMENT_FILTER_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

namespace kernel {
  using namespace ntt;

  template <Dimension D, unsigned short N>
  class MomentFilter_kernel {
    ndfield_t<D, N>       array;
    const ndfield_t<D, N> buffer;
    const unsigned short  comp_min, comp_max;

  public:
    MomentFilter_kernel(ndfield_t<D, N>&       array,
                        const ndfield_t<D, N>& buffer,
                        unsigned short         comp_min,
                        unsigned short         comp_max)
      : array { array }
      , buffer { buffer }
      , comp_min { comp_min }
      , comp_max { comp_max } {
      raise::ErrorIf(comp_min >= N || comp_max > N || comp_min >= comp_max,
                     "MomentFilter_kernel: invalid component range",
                     HERE);
    }

    Inline void operator()(index_t i1) const {
      if constexpr (D == Dim::_1D) {
        for (auto c = comp_min; c < comp_max; ++c) {
          array(i1, c) = INV_2 * buffer(i1, c) +
                         INV_4 *
                           (buffer(i1 - 1, c) + buffer(i1 + 1, c));
        }
      } else {
        raise::KernelError(
          HERE,
          "1D implementation of MomentFilter_kernel called for non-1D");
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim::_2D) {
        for (auto c = comp_min; c < comp_max; ++c) {
          array(i1, i2, c) =
            INV_4 * buffer(i1, i2, c) +
            INV_8 *
              (buffer(i1 - 1, i2, c) + buffer(i1 + 1, i2, c) +
               buffer(i1, i2 - 1, c) + buffer(i1, i2 + 1, c)) +
            INV_16 *
              (buffer(i1 - 1, i2 - 1, c) + buffer(i1 + 1, i2 + 1, c) +
               buffer(i1 - 1, i2 + 1, c) + buffer(i1 + 1, i2 - 1, c));
        }
      } else {
        raise::KernelError(
          HERE,
          "2D implementation of MomentFilter_kernel called for non-2D");
      }
    }

    Inline void operator()(index_t i1, index_t i2, index_t i3) const {
      if constexpr (D == Dim::_3D) {
        for (auto c = comp_min; c < comp_max; ++c) {
          // separable binomial: (1/4) * (1, 2, 1) along each axis
          // pre-compute the two intermediate axes' partial sums to keep
          // the arithmetic dense.
          real_t acc = ZERO;
          for (int dk = -1; dk <= 1; ++dk) {
            const real_t wk = (dk == 0) ? HALF : INV_4;
            for (int dj = -1; dj <= 1; ++dj) {
              const real_t wj = (dj == 0) ? HALF : INV_4;
              for (int di = -1; di <= 1; ++di) {
                const real_t wi = (di == 0) ? HALF : INV_4;
                acc += wi * wj * wk *
                       buffer(i1 + di, i2 + dj, i3 + dk, c);
              }
            }
          }
          array(i1, i2, i3, c) = acc;
        }
      } else {
        raise::KernelError(
          HERE,
          "3D implementation of MomentFilter_kernel called for non-3D");
      }
    }
  };

} // namespace kernel

#endif // KERNELS_MOMENT_FILTER_HPP
