using DataDrivenDiffEq
using DataDrivenDMD
using LinearAlgebra
using OrdinaryDiffEq

A_true = [-0.5 0.1; 0.0 -0.3]
B_true = [1.0; 0.5]

function control_signal(t)
    return sin(0.5 * t) + 0.3 * cos(1.2 * t)
end

function controlled_system!(du, u, p, t)
    ctrl = control_signal(t)
    return du .= A_true * u .+ B_true .* ctrl
end

u0 = [1.0, -0.5]
tspan = (0.0, 20.0)
dt = 0.1  # Sampling interval

prob = ODEProblem(controlled_system!, u0, tspan)
sol = solve(prob, Tsit5(), saveat = dt)

t_data = sol.t

X_data = Array(sol)

U_data = [control_signal(ti) for ti in t_data]
U_data = reshape(U_data, 1, :)  # Shape: (n_controls, n_timepoints)

ddprob = ContinuousDataDrivenProblem(X_data, t_data, U = U_data)

res = solve(ddprob, DMDSVD(), digits = 2)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
