
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)


function initial_condition_riemann_case2(x, t, equations::CompressibleEulerEquations2D)
  # change discontinuity to tanh
  # typical resolution 128^2, 256^2
  # domain size is [0.3, 0.6]^2
  if x[1] <= 0.5 && x[2] <= 0.5
    rho = 0.8
    v1 = 0.0
    v2 = 0.0
    p = 1.0
  elseif x[1] <= 0.5 && x[2] > 0.5
    rho = 1.0
    v1 = 0.7276
    v2 = 0.0
    p = 1.0
    u_prim = [1.0, 0.7276, 0.0, 1.0]
  elseif x[1] > 0.5 && x[2] <= 0.5
    rho = 1.0
    v1 = 0.0
    v2 = 0.7276
    p = 1.0
  elseif x[1] > 0.5 && x[2] > 0.5
    rho = 0.5313
    v1 = 0.0
    v2 = 0.0
    p = 0.4
  end
  
  return prim2cons(SVector(rho, v1, v2, p), equations)
end

#initial_condition = initial_condition_kelvin_helmholtz_instability
initial_condition = initial_condition_riemann_case2

surface_flux = flux_lax_friedrichs
volume_flux  = flux_ranocha
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.002,
                                         alpha_min=0.0001,
                                         alpha_smooth=true,
                                         variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

boundary_conditions = (
                        x_neg=BoundaryConditionDirichlet(initial_condition),
                        x_pos=BoundaryConditionDirichlet(initial_condition),
                        y_neg=BoundaryConditionDirichlet(initial_condition),
                        y_pos=BoundaryConditionDirichlet(initial_condition),
                       )

coordinates_min = ( 0.3, 0.3)
coordinates_max = ( 0.6, 0.6)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=5,
                periodicity=(false,false),
                n_cells_max=100_000)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.25)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=50,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=0.8)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary


pd_rc2_amr = PlotData2D(sol)

#using CairoMakie

# Plots
xs = range(0.3, 0.6, length = size(pd_rc2_amr.data[1])[1])
ys = range(0.3, 0.6, length = size(pd_rc2_amr.data[1])[2])

heatmap(xs, ys, pd_rc2_amr.data[1], title = string("Dichte ρ"), xlabel="x", ylabel="y",
        size = (600, 500))
