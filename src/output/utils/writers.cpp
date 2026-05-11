#include "output/utils/writers.h"

#include "global.h"

#include "arch/kokkos_aliases.h"

#include <adios2.h>

#include <cstddef>
#include <string>
#include <utility>

namespace out {

  // Pin every variable to MemorySpace::Host before each Put. We always hand
  // ADIOS2 a HostSpace pointer (mirror, std::vector data, stack scalar), but
  // ADIOS2's default `Detect` heuristic mis-classifies plain host VAs as GPU
  // on Aurora SYCL — BP5 then dispatches its `GPUMinMax<float>` stats kernel
  // onto an unmapped page and the L0 driver kills the run with a "Read
  // NotPresent" page fault (drm_neo.cpp:288). Detect is sticky on a
  // VariableBase, so we set it on every Put rather than once at Declare.
  template <typename T>
  static inline void pin_host(adios2::Variable<T>& var) {
    var.SetMemorySpace(adios2::MemorySpace::Host);
  }

  template <typename T>
  void WriteVariable(adios2::IO&        io,
                     adios2::Engine&    writer,
                     const std::string& name,
                     const T&           data,
                     std::size_t        global_size,
                     std::size_t        local_offset) {
    auto var = io.InquireVariable<T>(name);
    pin_host(var);
    var.SetShape({ global_size });
    var.SetSelection(adios2::Box<adios2::Dims>({ local_offset }, { 1 }));
    // Sync mode: ADIOS2 must consume `&data` before we return, since most
    // callers pass an rvalue (e.g. npart(), n_active()[d], extent()[d].first)
    // whose backing storage dies as soon as this call returns.
    writer.Put(var, &data, adios2::Mode::Sync);
  }

  template <typename T>
  void Write1DArray(adios2::IO&        io,
                    adios2::Engine&    writer,
                    const std::string& name,
                    const array_t<T*>& data,
                    std::size_t        local_size,
                    std::size_t        global_size,
                    std::size_t        local_offset) {
    const auto slice = std::pair<size_t, size_t>(0, local_size);
    auto       var   = io.InquireVariable<T>(name);
    pin_host(var);
    var.SetShape({ global_size });
    var.SetSelection(adios2::Box<adios2::Dims>({ local_offset }, { local_size }));

    // Mirror only the active range. For particle arrays `data.extent(0)` is
    // `maxnpart` (often >> local_size); creating a full-size host mirror and
    // deep-copying it would move hundreds of MB of unused tail per array on
    // every Put, blowing through PCIe bandwidth on first checkpoint and (on
    // Aurora) tripping GPU memory faults on the oversized async copies.
    auto data_d_sub = Kokkos::subview(data, slice);
    auto data_h     = Kokkos::create_mirror_view(data_d_sub);
    Kokkos::deep_copy(data_h, data_d_sub);
    writer.Put(var, data_h.data(), adios2::Mode::Sync);
  }

  template <typename T>
  void Write2DArray(adios2::IO&         io,
                    adios2::Engine&     writer,
                    const std::string&  name,
                    const array_t<T**>& data,
                    unsigned short      dim2_size,
                    std::size_t         local_size,
                    std::size_t         global_size,
                    std::size_t         local_offset) {
    const auto slice = std::pair<size_t, size_t>(0, local_size);
    auto       var   = io.InquireVariable<T>(name);
    pin_host(var);

    var.SetShape({ global_size * dim2_size });
    var.SetSelection(adios2::Box<adios2::Dims>({ local_offset * dim2_size },
                                               { local_size * dim2_size }));

    // Mirror only the active range (see Write1DArray comment).
    auto data_d_sub = Kokkos::subview(data,
                                      slice,
                                      std::pair<size_t, size_t>(0, dim2_size));
    auto data_h     = Kokkos::create_mirror_view(data_d_sub);
    Kokkos::deep_copy(data_h, data_d_sub);
    if (!data_h.span_is_contiguous()) {
      const Kokkos::View<T**, Kokkos::LayoutLeft, Kokkos::HostSpace>
        data_contig_h { "data_contig_h", local_size, dim2_size };
      Kokkos::deep_copy(data_contig_h, data_h);
      writer.Put(var, data_contig_h.data(), adios2::Mode::Sync);
    } else {
      writer.Put(var, data_h.data(), adios2::Mode::Sync);
    }
  }

  template <Dimension D, int N>
  void WriteNDField(adios2::IO&                      io,
                    adios2::Engine&                  writer,
                    const std::string&               name,
                    const ndfield_t<D, N>&           data,
                    const adios2::Box<adios2::Dims>& range) {
    auto var = io.InquireVariable<real_t>(name);
    pin_host(var);
    if (not range.first.empty()) {
      var.SetSelection(range);
    }
    auto data_h = Kokkos::create_mirror_view(data);
    Kokkos::deep_copy(data_h, data);
    writer.Put(var, data_h.data(), adios2::Mode::Sync);
  }

  // NOLINTBEGIN(bugprone-macro-parentheses)
#define ARRAY_WRITERS(T)                                                       \
  template void WriteVariable(adios2::IO&,                                     \
                              adios2::Engine&,                                 \
                              const std::string&,                              \
                              const T&,                                        \
                              std::size_t,                                     \
                              std::size_t);                                    \
  template void Write1DArray<T>(adios2::IO&,                                   \
                                adios2::Engine&,                               \
                                const std::string&,                            \
                                const array_t<T*>&,                            \
                                std::size_t,                                   \
                                std::size_t,                                   \
                                std::size_t);                                  \
  template void Write2DArray<T>(adios2::IO&,                                   \
                                adios2::Engine&,                               \
                                const std::string&,                            \
                                const array_t<T**>&,                           \
                                unsigned short,                                \
                                std::size_t,                                   \
                                std::size_t,                                   \
                                std::size_t);

  ARRAY_WRITERS(short)
  ARRAY_WRITERS(unsigned short)
  ARRAY_WRITERS(int)
  ARRAY_WRITERS(unsigned int)
  ARRAY_WRITERS(unsigned long int)
  ARRAY_WRITERS(double)
  ARRAY_WRITERS(float)
#undef ARRAY_WRITERS

#define NDFIELD_WRITERS(D, N)                                                  \
  template void WriteNDField<D, N>(adios2::IO&,                                \
                                   adios2::Engine&,                            \
                                   const std::string&,                         \
                                   const ndfield_t<D, N>&,                     \
                                   const adios2::Box<adios2::Dims>&);
  NDFIELD_WRITERS(Dim::_1D, 3)
  NDFIELD_WRITERS(Dim::_1D, 6)
  NDFIELD_WRITERS(Dim::_2D, 3)
  NDFIELD_WRITERS(Dim::_2D, 6)
  NDFIELD_WRITERS(Dim::_3D, 3)
  NDFIELD_WRITERS(Dim::_3D, 6)
#undef NDFIELD_WRITERS
  // NOLINTEND(bugprone-macro-parentheses)

} // namespace out
