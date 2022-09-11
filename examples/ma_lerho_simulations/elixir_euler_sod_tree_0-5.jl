
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations1D(1.4)


function initial_condition_sod_problem(x, t, equations::CompressibleEulerEquations1D)
  
  if x[1] <= 0
    rho = 1
    v1 = 0
    p = 1
  else
    rho = 0.125
    v1 = 0
    p = 0.1
  end

  return prim2cons(SVector(rho, v1, p), equations)
end

initial_condition = initial_condition_sod_problem

boundary_condition = initial_condition_sod_problem

boundary_conditions = (x_neg=BoundaryConditionDirichlet(boundary_condition),
                       x_pos=BoundaryConditionDirichlet(boundary_condition))

surface_flux = flux_lax_friedrichs
volume_flux  = flux_chandrashekar
basis = LobattoLegendreBasis(3)
shock_indicator_variable = density_pressure
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=shock_indicator_variable)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-0.5,)
coordinates_max = ( 0.5,)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=5,
                periodicity=(false),
                n_cells_max=10_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.14) 
ode = semidiscretize(semi, tspan)


summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max=0.5,
                                          alpha_min=0.001,
                                          alpha_smooth=true,
                                          variable=density_pressure)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=4,
                                      max_level=6, max_threshold=0.01)
#amr_callback = AMRCallback(semi, amr_controller,
#                           interval=5,
#                           adapt_initial_condition=true,
#                           adapt_initial_condition_only_refine=true)

stepsize_callback = StepsizeCallback(cfl=0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
#                        amr_callback,
                        stepsize_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=stepsize_callback(ode), # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary


