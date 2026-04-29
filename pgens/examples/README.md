# Examples

Problem generators in this directory are just examples demonstrating how to use some of the features of the Entity.

- `custom_energy_distribution`: demonstrates how to initialize a uniform distribution of particles with two different custom-defined energy (velocity) distributions (per each species).

  <img width="1024" alt="image" src="https://github.com/user-attachments/assets/d8499ab7-31dc-49e8-8381-34aa8a994be0" />

- `custom_spatial_distribution`: example of using the non-uniform plasma injector with a custom spatial distirbution.

  https://github.com/user-attachments/assets/bea0c290-e7e4-4ec7-b360-68ce76beab5b

- `match_fix_field_boundaries`: example of setting matching and/or fixed (coordinate-independent) field boundaries for the electromagnetic fields.

  https://github.com/user-attachments/assets/a1b9ea22-34ce-474f-a9b0-49789a2e52b3

- `custom_emission`: simple example where two particles are initialized on gyrating trajectories probabilistically emitting photons while the total energy is conserved.

  https://github.com/user-attachments/assets/6c5c399a-bbab-4b93-be5f-35fed1959fcc

- `external_fields`: example of using external fields for supplying a time-varying and species-dependent magnetic/electric fields and/or force-field imposed on particles.

  <img width="512" alt="image" src="https://github.com/user-attachments/assets/7126fcdb-7484-4695-86c1-cdd425cc4655" />

- `atmosphere`: setting up a gravitationally bound "atmosphere" with a constant particle replenisher (the plot also highlights the importance of having a constant replenisher and a gravity force acting on the particles; both are enabled by default when using the `ATMOSPHERE` particle boundary conditions).

  <img width="1024" alt="atmosphere" src="https://github.com/user-attachments/assets/70ac2cff-6775-47a3-a394-bf6df2533344" />

- `replenish_injector`: demonstration of how to use the replenish injector to periodically inject new plasma to a target density (both uniform and non-uniform) in `CustomPostStep`.

  https://github.com/user-attachments/assets/4955f05e-4795-439f-a4a5-8196e42e987b
  
  https://github.com/user-attachments/assets/ebc303ed-7b21-4f97-9822-d81124fdf962

- `piston`: implementing a moving piston which continuously pushes the plasma.

  https://github.com/user-attachments/assets/7d2dfc39-5de4-49cb-9d55-be8700cbcaee

- `moving_window`: demonstrates the domain being updated to follow certain features in your simulation in one of the direction (jerkiness is due to updates taking place every `n`-th timestep, if you limit plotting to a specific physical interval, this goes away).

  https://github.com/user-attachments/assets/00edb648-af80-4d37-b1c2-c49872227d47

- `ascent_cube`: minimal example demonstrating the in-situ visualization interface to [Ascent](https://ascent.readthedocs.io/). Renders a 3D pseudocolor plot of the `B3` component on a periodic cube every output cycle. Requires building Entity with `-D ascent=ON` (Ascent must be installed and discoverable via `Ascent_DIR`).

- `turbulence_ascent`: 3D driven turbulence (re-using `pgens/turbulence`) with an Ascent pipeline that composites `B1, B2, B3` into a vector field, derives `|B|`, and ray-traces a volume render every render cycle. Requires `-D ascent=ON`.

- `turbulence_lines`: same 3D turbulence setup as `turbulence_ascent`, but the Ascent pipeline traces magnetic field lines with the `streamline` filter from a uniform grid of seeds and rasterizes them as 3D tubes. Requires `-D ascent=ON`.

- `turbulence_density_lines`: combines the previous two — a ray-traced volume of the particle density `N` overlaid with magnetic field lines, both composited into one PNG per render cycle. Requires `-D ascent=ON`.



