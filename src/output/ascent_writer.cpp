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
  #include <conduit_relay.hpp>

  #if defined(MPI_ENABLED)
    #include <mpi.h>
  #endif

  #include <cstdint>
  #include <filesystem>
  #include <fstream>
  #include <sstream>
  #include <string>
  #include <vector>

namespace out {

  namespace {
    /*
     * Conduit's YAML parser stores integer scalars as int64 by default, but
     * many Ascent / VTK-h filters call `.as_int()` (i.e. int32) on their
     * parameters and that check is strict. Walk the parsed actions tree
     * once and demote int64 leaves to int32 when the value fits — this is
     * safe for the small counters used in actions files (num_steps,
     * num_seeds_*, image_width, ...).
     *
     * We also rewrite multi-element int64 arrays into int32 arrays so that
     * filters like `point_list` whose params are int-vectors stay valid.
     *
     * Returns the number of leaves that were demoted.
     */
    auto demoteInt64ToInt32(conduit::Node& node) -> std::size_t {
      std::size_t n_demoted = 0;
      // Catch any integer leaf that isn't already int32. We use
      // `.to_int64()` so we don't have to know the source type — it does
      // a value-preserving cast for int8/16/32/64 and uint8/16/32/64.
      if (node.dtype().is_integer() and not node.dtype().is_int32()) {
        const auto nelem = node.dtype().number_of_elements();
        if (nelem == 1) {
          const conduit::int64 v = node.to_int64();
          if (v >= static_cast<conduit::int64>(INT32_MIN) and
              v <= static_cast<conduit::int64>(INT32_MAX)) {
            node.set_int32(static_cast<conduit::int32>(v));
            ++n_demoted;
          }
        } else if (nelem > 1) {
          // multi-element int array → int32 array (if all values fit)
          conduit::int64_array src = node.as_int64_array();
          std::vector<conduit::int32> tmp;
          tmp.reserve(static_cast<std::size_t>(nelem));
          bool fits = true;
          for (conduit::index_t i = 0; i < nelem; ++i) {
            const conduit::int64 v = src[i];
            if (v < static_cast<conduit::int64>(INT32_MIN) or
                v > static_cast<conduit::int64>(INT32_MAX)) {
              fits = false;
              break;
            }
            tmp.push_back(static_cast<conduit::int32>(v));
          }
          if (fits) {
            node.set(tmp.data(), tmp.size());
            ++n_demoted;
          }
        }
      }
      for (conduit::index_t i = 0; i < node.number_of_children(); ++i) {
        n_demoted += demoteInt64ToInt32(node.child(i));
      }
      return n_demoted;
    }
  } // namespace

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
                          simtime_t                       interval_time,
                          bool                            vector_aliases) {
    raise::ErrorIf(m_initialized, "AscentWriter already initialized", HERE);

    m_root           = title;
    m_actions_file   = actions_file;
    m_fields         = fields;
    m_vector_aliases = vector_aliases;

    m_tracker.init("ascent", interval, interval_time);

    // Mirror the layout used by the ADIOS field/particle writers:
    // <sim_name>/<sub>/<files>. Renders go under <sim_name>/plots/.
    const auto plots_dir = std::filesystem::path(m_root) / "plots";
    std::filesystem::create_directories(plots_dir);

    m_options.reset();
  #if defined(MPI_ENABLED)
    m_options["mpi_comm"] = MPI_Comm_c2f(MPI_COMM_WORLD);
  #endif
    m_options["default_dir"] = plots_dir.string();
    m_options["exceptions"]  = "forward";
    m_options["messages"]    = "quiet";

    // Pre-parse the user's actions file so we can demote int64 leaves to
    // int32 (Conduit YAML defaults to int64 but many Ascent filters call
    // `.as_int()` strictly). Only when this fails do we fall back to
    // letting Ascent read the file itself.
    m_actions.reset();
    m_have_actions = false;
    if (!m_actions_file.empty()) {
      try {
        if (!std::filesystem::exists(m_actions_file)) {
          throw std::runtime_error("actions file not found at '" +
                                   m_actions_file + "' (cwd: " +
                                   std::filesystem::current_path().string() +
                                   ")");
        }
        // Use the same loader Ascent uses internally so the parse is
        // guaranteed compatible with the YAML schema Ascent expects.
        conduit::relay::io::load(m_actions_file, "yaml", m_actions);
        const auto n_demoted = demoteInt64ToInt32(m_actions);
        m_have_actions = true;
        logger::Checkpoint(
          "Pre-loaded Ascent actions file '" + m_actions_file + "' (demoted " +
            std::to_string(n_demoted) + " int64 -> int32 leaves)",
          HERE);
        // CRITICAL: Ascent's `execute()` always calls CheckForSettingsFile
        // with merge=false, which REPLACES our in-memory actions with a
        // freshly-parsed copy from disk if `actions_file` resolves to a
        // file that exists. By default `actions_file` is unset, in which
        // case Ascent falls back to "ascent_actions.yaml" in the cwd.
        // Set actions_file to "" so `is_file("")` is false and the
        // disk-load path (which would discard our int32 demotion) is
        // skipped.
        m_options["actions_file"] = "";
        // Drop a copy of the patched actions inside the plots directory
        // so users can inspect exactly what was sent to Ascent.
        try {
          const auto settings_path = plots_dir / "ascent_settings.yaml";
          conduit::relay::io::save(m_actions, settings_path.string(), "yaml");
        } catch (...) {
          // best-effort debug dump; don't crash if it fails
        }
      } catch (const std::exception& e) {
        raise::Warning(
          "Could not pre-load Ascent actions file '" + m_actions_file +
            "' (" + e.what() +
            "); falling back to Ascent's own loader (int32 demotion skipped).",
          HERE);
        m_options["actions_file"] = m_actions_file;
      }
    }

    m_ascent.open(m_options);
    m_initialized = true;

    // Surface the active Ascent runtime + backend so users can confirm
    // they're getting the GPU-accelerated path. If the build accidentally
    // fell back to the serial host runtime, rendering 10^7+ cells will be
    // dominated by VTK-m on the CPU.
    //
    // `ascent::about(node)` is a free function (not a method on the
    // ascent::Ascent instance) and reports build-time configuration:
    // version, the default runtime, and per-runtime availability
    // including the active VTK-m backend. The exact key layout varies
    // between Ascent versions, so we probe a handful of well-known paths
    // and fall back to dumping a YAML summary if none match.
    try {
      conduit::Node about_node;
      ::ascent::about(about_node);

      // Dump the full about node so users can read the exact key layout
      // for the installed Ascent version (paths drift between releases).
      try {
        const auto about_path = plots_dir / "ascent_about.yaml";
        conduit::relay::io::save(about_node, about_path.string(), "yaml");
      } catch (...) {
        // best-effort debug dump
      }

      std::string summary = "Ascent";
      if (about_node.has_path("version")) {
        summary += " v" + about_node["version"].as_string();
      }
      std::string default_runtime;
      if (about_node.has_path("default_runtime")) {
        default_runtime = about_node["default_runtime"].as_string();
        summary += " runtime=" + default_runtime;
      }

      // Enumerate VTK-m backends marked "enabled" in the about node.
      // Layout in Ascent 0.9.x:
      //   runtimes/<default_runtime>/vtkm/backends/<name> = "enabled"|"disabled"
      // This reports what was BUILT into VTK-m. The active dispatcher
      // when the Kokkos backend is selected is determined by Kokkos's
      // default execution space (printed below).
      if (!default_runtime.empty()) {
        const std::string backends_path = "runtimes/" + default_runtime +
                                          "/vtkm/backends";
        if (about_node.has_path(backends_path)) {
          const auto& backends = about_node[backends_path];
          std::string enabled_list;
          for (conduit::index_t i = 0; i < backends.number_of_children();
               ++i) {
            const auto& c = backends.child(i);
            if (c.dtype().is_string() && c.as_string() == "enabled") {
              if (!enabled_list.empty()) {
                enabled_list += ",";
              }
              enabled_list += c.name();
            }
          }
          if (!enabled_list.empty()) {
            summary += " vtkm-backends=" + enabled_list;
          }
        }
      }

      // Kokkos's default execution space is the actual dispatcher when
      // Ascent runs through the Kokkos VTK-m backend. This name (SYCL /
      // CUDA / HIP / OpenMP / Serial) is the source of truth for whether
      // in situ rendering runs on the GPU.
      summary += " kokkos=";
      summary += Kokkos::DefaultExecutionSpace::name();

      logger::Checkpoint(summary, HERE);
    } catch (...) {
      // never let an introspection failure abort init
    }
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
                                const std::vector<std::size_t>& l_shape,
                                const std::vector<std::size_t>& l_first_cell,
                                const std::vector<std::size_t>& downsample) {
    raise::ErrorIf(!m_initialized, "AscentWriter not initialized", HERE);
    raise::ErrorIf(l_corner.size() != static_cast<std::size_t>(dim) ||
                     l_shape.size() != static_cast<std::size_t>(dim),
                   "AscentWriter::defineMesh size mismatch",
                   HERE);
    raise::ErrorIf(!l_first_cell.empty() &&
                     l_first_cell.size() != static_cast<std::size_t>(dim),
                   "AscentWriter::defineMesh l_first_cell size mismatch",
                   HERE);
    raise::ErrorIf(!downsample.empty() &&
                     downsample.size() != static_cast<std::size_t>(dim),
                   "AscentWriter::defineMesh downsample size mismatch",
                   HERE);
    m_dim      = dim;
    m_l_corner = l_corner;
    m_l_shape  = l_shape;

    m_l_first_cell.assign(static_cast<std::size_t>(dim), 0u);
    if (!l_first_cell.empty()) {
      m_l_first_cell = l_first_cell;
    }
    m_downsample.assign(static_cast<std::size_t>(dim), 1u);
    if (!downsample.empty()) {
      m_downsample = downsample;
      for (auto s : m_downsample) {
        raise::ErrorIf(s == 0u, "downsample factor must be nonzero", HERE);
      }
    }
    // Mesh structure is being (re)defined; force a fresh blueprint
    // verification on the next render.
    m_verified = false;

    m_mesh.reset();
    m_mesh["coordsets/coords/type"] = "rectilinear";
    // coordinate arrays are filled later via setMeshCoords()
    m_mesh["topologies/mesh/type"]     = "rectilinear";
    m_mesh["topologies/mesh/coordset"] = "coords";
    m_mesh_defined                     = true;

    // Allocate the per-field staging buffers once. Reused across every
    // publishField() call (and across renders) to avoid the per-call
    // device + host + std::vector allocation seen in earlier revisions.
    std::size_t nelem = 1;
    for (auto n : m_l_shape) {
      nelem *= n;
    }
    m_buf_nelem   = nelem;
    m_field_buf_d = array_t<double*>("ascent_field_buf", nelem);
    m_field_buf_h = Kokkos::create_mirror_view(m_field_buf_d);
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
    raise::ErrorIf(m_buf_nelem == 0,
                   "AscentWriter staging buffer not allocated",
                   HERE);

    const std::size_t gh  = ntt::N_GHOSTS;
    auto              buf = m_field_buf_d;

    // Extract the (possibly downsampled) active region of the requested
    // component into the persistent flat buffer in i-fastest order
    // (Conduit/Blueprint layout). Conduit's `set(...)` below copies out,
    // so the buffer can be reused across fields and across renders.
    //
    // Source index for downsampled cell `i_dwn` along axis d is
    //   i_src = m_l_first_cell[d] + i_dwn * m_downsample[d] + N_GHOSTS
    // which matches how `output.fields.downsampling` strides through the
    // ADIOS path.
    if constexpr (D == Dim::_3D) {
      const std::size_t n1 = m_l_shape[0];
      const std::size_t n2 = m_l_shape[1];
      const std::size_t n3 = m_l_shape[2];
      const std::size_t f1 = m_l_first_cell[0];
      const std::size_t f2 = m_l_first_cell[1];
      const std::size_t f3 = m_l_first_cell[2];
      const std::size_t s1 = m_downsample[0];
      const std::size_t s2 = m_downsample[1];
      const std::size_t s3 = m_downsample[2];
      Kokkos::parallel_for(
        "AscentExtract3D",
        CreateRangePolicy<Dim::_3D>({ 0, 0, 0 }, { n1, n2, n3 }),
        Lambda(index_t i1, index_t i2, index_t i3) {
          buf(i1 + n1 * (i2 + n2 * i3)) = static_cast<double>(
            fld(f1 + i1 * s1 + gh,
                f2 + i2 * s2 + gh,
                f3 + i3 * s3 + gh,
                comp));
        });
    } else if constexpr (D == Dim::_2D) {
      const std::size_t n1 = m_l_shape[0];
      const std::size_t n2 = m_l_shape[1];
      const std::size_t f1 = m_l_first_cell[0];
      const std::size_t f2 = m_l_first_cell[1];
      const std::size_t s1 = m_downsample[0];
      const std::size_t s2 = m_downsample[1];
      Kokkos::parallel_for(
        "AscentExtract2D",
        CreateRangePolicy<Dim::_2D>({ 0, 0 }, { n1, n2 }),
        Lambda(index_t i1, index_t i2) {
          buf(i1 + n1 * i2) = static_cast<double>(
            fld(f1 + i1 * s1 + gh, f2 + i2 * s2 + gh, comp));
        });
    } else { // Dim::_1D
      const std::size_t n1 = m_l_shape[0];
      const std::size_t f1 = m_l_first_cell[0];
      const std::size_t s1 = m_downsample[0];
      Kokkos::parallel_for(
        "AscentExtract1D",
        n1,
        Lambda(index_t i1) {
          buf(i1) = static_cast<double>(fld(f1 + i1 * s1 + gh, comp));
        });
    }
    Kokkos::deep_copy(m_field_buf_h, m_field_buf_d);

    // Internal field names carry a leading "f" prefix (see out::OutputField).
    // The user-facing name in Ascent / Conduit is the same name without it.
    const std::string short_name = (name.size() > 1u && name.front() == 'f')
                                     ? name.substr(1)
                                     : name;
    const std::string base = "fields/" + short_name;
    m_mesh[base + "/topology"]    = "mesh";
    m_mesh[base + "/association"] = "element";
    m_mesh[base + "/values"].set(m_field_buf_h.data(), m_buf_nelem);

    // Vector-field bonus path (gated by m_vector_aliases): when the
    // user-facing name ends in "1", "2", or "3" (e.g. B1/B2/B3) ALSO
    // publish the same values as the matching component of an MCArray
    // vector field whose name is the prefix (e.g. "B"). VTK-h filters
    // that need a 3-vector (streamline, gradient, ...) can then
    // reference the prefix name directly without an extra
    // `composite_vector` pipeline step — that workflow currently
    // produces a vector layout VTK-h's streamline backend in Ascent
    // v0.9.x can't unpack as Vec<float,3>/Vec<double,3>.
    // Disable via `output.ascent.vector_aliases = false` in the toml
    // input to halve published mesh size when the actions file only
    // references the scalar form (or only the vector form).
    if (m_vector_aliases && short_name.size() >= 2u) {
      const char idx = short_name.back();
      if (idx >= '1' && idx <= '3') {
        const std::string prefix = short_name.substr(0, short_name.size() - 1u);
        if (!prefix.empty()) {
          static const char* const sub[] = { "u", "v", "w" };
          const std::string vec_base = "fields/" + prefix;
          m_mesh[vec_base + "/topology"]    = "mesh";
          m_mesh[vec_base + "/association"] = "element";
          m_mesh[vec_base + "/values/" + sub[idx - '1']].set(
            m_field_buf_h.data(),
            m_buf_nelem);
        }
      }
    }
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

    // Mesh structure (coordsets, topologies, field names/associations) is
    // identical from one render to the next — only state and field values
    // change. Run the full Blueprint verify once; on subsequent renders it
    // is pure overhead. `defineMesh` resets `m_verified` if the layout is
    // re-declared.
    if (!m_verified) {
      conduit::Node verify_info;
      if (!conduit::blueprint::mesh::verify(m_mesh, verify_info)) {
        raise::Warning("Ascent blueprint verification failed: " +
                         verify_info.to_yaml(),
                       HERE);
        m_pending_render = false;
        return false;
      }
      m_verified = true;
    }

    m_ascent.publish(m_mesh);

    // When we successfully pre-parsed the actions file in init(), pass it
    // to execute() directly (with int64→int32 demotion already applied).
    // Otherwise fall back to an empty node, so Ascent reads
    // ascent_actions.yaml itself via the `actions_file` option.
    if (m_have_actions) {
      m_ascent.execute(m_actions);
    } else {
      conduit::Node actions;
      m_ascent.execute(actions);
    }
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
