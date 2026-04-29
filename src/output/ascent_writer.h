/**
 * @file output/ascent_writer.h
 * @brief Wrapper around the Ascent in situ visualization library
 * @implements
 *   - out::AscentWriter
 * @cpp:
 *   - ascent_writer.cpp
 * @namespaces:
 *   - out::
 * @macros:
 *   - MPI_ENABLED
 *   - OUTPUT_ENABLED
 *   - ASCENT_ENABLED
 * @note
 * AscentWriter mirrors the layout of out::Writer but instead of dumping data
 * to ADIOS2 it publishes a Conduit-Blueprint mesh to Ascent and triggers a
 * user-supplied actions file to render images on the fly.
 */

#ifndef OUTPUT_ASCENT_WRITER_H
#define OUTPUT_ASCENT_WRITER_H

#if defined(ASCENT_ENABLED)

  #include "enums.h"
  #include "global.h"

  #include "arch/kokkos_aliases.h"
  #include "utils/tools.h"

  #include <ascent.hpp>
  #include <conduit.hpp>

  #if defined(MPI_ENABLED)
    #include <mpi.h>
  #endif

  #include <string>
  #include <vector>

namespace out {

  class AscentWriter {
    ascent::Ascent m_ascent;
    conduit::Node  m_mesh;
    conduit::Node  m_options;
    bool           m_initialized { false };
    bool           m_mesh_defined { false };
    bool           m_pending_render { false };

    Dimension                m_dim { Dim::_3D };
    std::vector<std::size_t> m_l_shape;
    std::vector<std::size_t> m_l_corner;
    std::string              m_root;
    std::string              m_actions_file;
    std::vector<std::string> m_fields;

    tools::Tracker m_tracker;

  public:
    AscentWriter() = default;
    ~AscentWriter();

    AscentWriter(const AscentWriter&)            = delete;
    AscentWriter& operator=(const AscentWriter&) = delete;

    /**
     * @brief Initialize the underlying ascent::Ascent instance.
     * @param title Simulation name (used for the output directory).
     * @param actions_file Path to a yaml/json file with Ascent actions.
     * @param fields List of field names (e.g. "B3") that will be published.
     * @param interval Step interval between renders (used when interval_time<=0).
     * @param interval_time Sim-time interval between renders.
     */
    void init(const std::string&              title,
              const std::string&              actions_file,
              const std::vector<std::string>& fields,
              timestep_t                      interval,
              simtime_t                       interval_time);

    /**
     * @brief Whether the writer should fire on the current cycle.
     */
    auto shouldRender(timestep_t step, simtime_t time) -> bool;

    /**
     * @brief Define the local rectilinear-mesh layout.
     * @param dim Dimensionality of the mesh (1/2/3).
     * @param l_corner Local lower-left corner in global cell index space.
     * @param l_shape Local number of active cells in each direction.
     */
    void defineMesh(Dimension                       dim,
                    const std::vector<std::size_t>& l_corner,
                    const std::vector<std::size_t>& l_shape);

    /**
     * @brief Set the cell-edge coordinate arrays for one dimension.
     */
    void setMeshCoords(unsigned short dim, const array_t<real_t*>& xe);

    /**
     * @brief Push a single field component into the mesh blueprint.
     * @param name Field name (used by the actions file).
     * @param fld Backing storage for the simulation fields.
     * @param comp Component index to extract.
     */
    template <Dimension D, int N>
    void publishField(const std::string&     name,
                      const ndfield_t<D, N>& fld,
                      std::size_t            comp);

    /**
     * @brief Trigger the Ascent pipeline for the current step.
     * @return true if the pipeline actually executed (data was pending),
     *         false if there was nothing to render.
     */
    auto render(timestep_t step, simtime_t time) -> bool;

    /**
     * @brief Whether there is data published since the last render.
     */
    [[nodiscard]]
    auto hasPending() const -> bool {
      return m_pending_render;
    }

    /**
     * @brief Whether the writer was successfully initialized.
     */
    [[nodiscard]]
    auto initialized() const -> bool {
      return m_initialized;
    }

    /**
     * @brief Names of fields registered for in situ rendering.
     */
    [[nodiscard]]
    auto fields() const -> const std::vector<std::string>& {
      return m_fields;
    }
  };

} // namespace out

#endif // ASCENT_ENABLED

#endif // OUTPUT_ASCENT_WRITER_H
