using Trixi, OrdinaryDiffEq

surface_flux = FluxLaxFriedrichs()
volume_flux = flux_ranocha
dg = DGMulti(polydeg = 3, element_type = Line(), approximation_type = Polynomial(),
             surface_integral = SurfaceIntegralWeakForm(surface_flux),
             volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

equations = CompressibleEulerEquations1D(1.4)

initial_condition = initial_condition_convergence_test
source_terms = source_terms_convergence_test

cells_per_dimension = (8, )
vertex_coordinates, EToV = StartUpDG.uniform_mesh(dg.basis.elementType, cells_per_dimension...)
mesh = VertexMappedMesh(vertex_coordinates, EToV, dg, is_periodic=(true,))
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg;
                                    source_terms=source_terms)

tspan = (0.0, 1.1)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, uEltype=real(dg))
callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt = 0.5 * estimate_dt(mesh, dg), save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary
