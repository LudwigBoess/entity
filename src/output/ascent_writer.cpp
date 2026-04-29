#include "output/ascent_writer.h"

#if defined(ASCENT_ENABLED)

  #include "enums.h"
  #include "global.h"

  #include "arch/kokkos_aliases.h"
  #include "utils/error.h"
  #include "utils/log.h"

  #include <Kokkos_Core.hpp>
  #include <ascent.hpp>
  #include <conduit.hpp>
  #include <conduit_blueprint.hpp>

  #if defined(MPI_ENABLED)
    #include <mpi.h>
  #endif

  #include <filesystem>
  #include <string>
  #include <vector>

namespace out {

  AscentWriter::~AscentWriter() {
    if (m_initialized) {
      try {
        m_ascent.close();
      } catch (...) {
        // swallow shutdown errors so we never throw from a destructor
      }
      m_initialized = false;
    }
  }

  void AscentWriter::init(const std::string&              title,
                          const std::string&              actions_file,
                          const std::vector<std::string>& fields,
                          timestep_t                      interval,
                          simtime_t                       interval_time) {
    raise::ErrorIf(m_initialized, "AscentWriter already initialized", HERE);

    m_root         = title;
    m_actions_file = actions_file;
    m_fields       = fields;

    m_tracker.init("ascent", interval, interval_time);

    if (!std::filesystem::exists(m_root)) {
      std::filesystem::create_directories(m_root);
    }

    m_options.reset();
  #if defined(MPI_ENABLED)
    m_options["mpi_comm"] = MPI_Comm_c2f(MPI_COMM_WORLD);
  #endif
    if (!m_actions_file.empty()) {
      m_options["actions_file"] = m_actions_file;
    }
    m_options["default_dir"] = m_root;
    m_options["exceptions"]  = "forward";
    m_options["messages"]    = "quiet";

    m_ascent.open(m_options);
    m_initialized = true;
    logger::Checkpoint("Initialized Ascent in situ writer", HERE);
  }

  auto AscentWriter::shouldRender(timestep_t step, simtime_t time) -> bool {
    if (!m_initialized) {
      return false;
    }
    return m_tracker.shouldWrite(step, time);
  }

  void AscentWriter::defineMesh(Dimension                       dim,
                                const std::vector<std::size_t>& l_corner,
                                const std::vector<std::size_t>& l_shape) {
    raise::ErrorIf(!m_initialized, "AscentWriter not initialized", HERE);
    raise::ErrorIf(l_corner.size() != static_cast<std::size_t>(dim) ||
                     l_shape.size() != static_cast<std::size_t>(dim),
                   "AscentWriter::defineMesh size mismatch",
                   HERE);
    m_dim      = dim;
    m_l_corner = l_corner;
    m_l_shape  = l_shape;

    m_mesh.reset();
    m_mesh["coordsets/coords/type"] = "rectilinear";
    // coordinate arrays are filled later via setMeshCoords()
    m_mesh["topologies/mesh/type"]     = "rectilinear";
    m_mesh["topologies/mesh/coordset"] = "coords";
    m_mesh_defined                     = true;
  }

  void AscentWriter::setMeshCoords(unsigned short          dim,
                                   const array_t<real_t*>& xe) {
    raise::ErrorIf(!m_mesh_defined, "AscentWriter mesh not defined", HERE);
    raise::ErrorIf(dim >= static_cast<unsigned short>(m_dim),
                   "AscentWriter::setMeshCoords invalid dim",
                   HERE);
    auto xe_h = Kokkos::create_mirror_view(xe);
    Kokkos::deep_copy(xe_h, xe);

    static const char* const axes[3] = { "x", "y", "z" };
    std::vector<double>      values(xe_h.extent(0));
    for (std::size_t i = 0; i < xe_h.extent(0); ++i) {
      values[i] = static_cast<double>(xe_h(i));
    }
    m_mesh["coordsets/coords/values/" + std::string(axes[dim])].set(
      values.data(),
      values.size());
  }

  template <Dimension D, int N>
  void AscentWriter::publishField(const std::string&     name,
                                  const ndfield_t<D, N>& fld,
                                  std::size_t            comp) {
    raise::ErrorIf(!m_mesh_defined, "AscentWriter mesh not defined", HERE);

    const std::size_t gh = ntt::N_GHOSTS;
    std::vector<double> values;

    if constexpr (D == Dim::_3D) {
      const std::size_t n1 = m_l_shape[0];
      const std::size_t n2 = m_l_shape[1];
      const std::size_t n3 = m_l_shape[2];
      values.resize(n1 * n2 * n3);

      ndarray_t<Dim::_3D> slice { "ascent_field", n1, n2, n3 };
      Kokkos::parallel_for(
        "AscentExtract3D",
        CreateRangePolicy<Dim::_3D>({ 0, 0, 0 }, { n1, n2, n3 }),
        Lambda(index_t i1, index_t i2, index_t i3) {
          slice(i1, i2, i3) = fld(i1 + gh, i2 + gh, i3 + gh, comp);
        });
      auto slice_h = Kokkos::create_mirror_view(slice);
      Kokkos::deep_copy(slice_h, slice);
      // Conduit/Blueprint expects logical-i fastest-varying for implicit topologies
      for (std::size_t k = 0; k < n3; ++k) {
        for (std::size_t j = 0; j < n2; ++j) {
          for (std::size_t i = 0; i < n1; ++i) {
            values[i + n1 * (j + n2 * k)] = static_cast<double>(slice_h(i, j, k));
          }
        }
      }
    } else if constexpr (D == Dim::_2D) {
      const std::size_t n1 = m_l_shape[0];
      const std::size_t n2 = m_l_shape[1];
      values.resize(n1 * n2);

      ndarray_t<Dim::_2D> slice { "ascent_field", n1, n2 };
      Kokkos::parallel_for(
        "AscentExtract2D",
        CreateRangePolicy<Dim::_2D>({ 0, 0 }, { n1, n2 }),
        Lambda(index_t i1, index_t i2) {
          slice(i1, i2) = fld(i1 + gh, i2 + gh, comp);
        });
      auto slice_h = Kokkos::create_mirror_view(slice);
      Kokkos::deep_copy(slice_h, slice);
      for (std::size_t j = 0; j < n2; ++j) {
        for (std::size_t i = 0; i < n1; ++i) {
          values[i + n1 * j] = static_cast<double>(slice_h(i, j));
        }
      }
    } else { // Dim::_1D
      const std::size_t n1 = m_l_shape[0];
      values.resize(n1);

      ndarray_t<Dim::_1D> slice { "ascent_field", n1 };
      Kokkos::parallel_for(
        "AscentExtract1D",
        n1,
        Lambda(index_t i1) { slice(i1) = fld(i1 + gh, comp); });
      auto slice_h = Kokkos::create_mirror_view(slice);
      Kokkos::deep_copy(slice_h, slice);
      for (std::size_t i = 0; i < n1; ++i) {
        values[i] = static_cast<double>(slice_h(i));
      }
    }

    // Internal field names carry a leading "f" prefix (see out::OutputField).
    // The user-facing name in Ascent / Conduit is the same name without it.
    const std::string short_name = (name.size() > 1u && name.front() == 'f')
                                     ? name.substr(1)
                                     : name;
    const std::string base = "fields/" + short_name;
    m_mesh[base + "/topology"]    = "mesh";
    m_mesh[base + "/association"] = "element";
    m_mesh[base + "/values"].set(values.data(), values.size());
    m_pending_render = true;
  }

  auto AscentWriter::render(timestep_t step, simtime_t time) -> bool {
    if (!m_initialized || !m_mesh_defined || !m_pending_render) {
      return false;
    }
    m_mesh["state/cycle"]     = static_cast<conduit::int64>(step);
    m_mesh["state/time"]      = static_cast<conduit::float64>(time);
    m_mesh["state/domain_id"] = 0;
  #if defined(MPI_ENABLED)
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    m_mesh["state/domain_id"] = rank;
  #endif

    conduit::Node verify_info;
    if (!conduit::blueprint::mesh::verify(m_mesh, verify_info)) {
      raise::Warning("Ascent blueprint verification failed: " +
                       verify_info.to_yaml(),
                     HERE);
      m_pending_render = false;
      return false;
    }

    m_ascent.publish(m_mesh);

    // Empty actions => Ascent reads from the actions file (or
    // ascent_actions.yaml in the current directory) and applies it.
    conduit::Node actions;
    m_ascent.execute(actions);
    m_pending_render = false;
    return true;
  }

  // Explicit instantiations matching the storage in the simulation.
  template void AscentWriter::publishField<Dim::_1D, 6>(const std::string&,
                                                        const ndfield_t<Dim::_1D, 6>&,
                                                        std::size_t);
  template void AscentWriter::publishField<Dim::_2D, 6>(const std::string&,
                                                        const ndfield_t<Dim::_2D, 6>&,
                                                        std::size_t);
  template void AscentWriter::publishField<Dim::_3D, 6>(const std::string&,
                                                        const ndfield_t<Dim::_3D, 6>&,
                                                        std::size_t);

} // namespace out

#endif // ASCENT_ENABLED
