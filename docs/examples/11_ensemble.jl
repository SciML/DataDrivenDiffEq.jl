
using Revise
using OrdinaryDiffEq
using Plots
using ModelingToolkit
using StatsBase
using LinearAlgebra
using Measurements
using DataDrivenDiffEq

function pendulum(u, p, t)
    x = u[2]
    y = -p[1]*sin(u[1]) - p[2]*u[2] - p[3]*u[1] 
    return [x;y]
end

u_init() = randn(2)

p_init() = begin
    p_1 = 9.81 * (1 .+ 0.1*(rand() - 0.5))
    p_2 = 0.5 * (1 .+ 0.1*(rand() - 0.5))
    p_3 = 0.2 * (1 + 0.1*(rand() - 0.5))
    vcat(p_1, p_2, p_3)
end

t_observed(tspan) = begin
    δ = -(-(tspan...) / 100)
    sort(vcat(first(tspan), sample(first(tspan):δ:last(tspan), rand(20:1:30), replace = false)))
end

tspan = (0.0, 10.0)
u0 = u_init()
p0 = p_init()

prob = ODEProblem(pendulum, u0, tspan, p0)
sol = solve(prob, Tsit5(), saveat = t_observed(tspan))
plot(sol)

solve_realization(prob) = begin
    solve(
        prob, Tsit5(), 
        u0 = u_init(), p = p_init(), saveat  = t_observed(prob.tspan)
    )
end

solutions = map(1:50) do i 
    solve_realization(prob)
end

pl = plot()
map(solutions) do s
    plot!(s, alpha = 0.4, label = nothing)
end
display(pl)

# Create an ensemble problem
ddprob = DataDrivenEnsemble(solutions...)

@variables u[1:2]
u = collect(u)

h = Num[sin.(u[1]);cos.(u[1]); polynomial_basis(u, 5)]

basis = Basis(h, u)

# To solve the problem, we also define a [`DataSampler`](@ref) which defines randomly shuffled minibatches of our data and selects the 
# best fit.

sampler = DataSampler(Batcher(n = 3, shuffle = true, repeated = true))
λs = exp10.(-10:0.1:-1)
opt = STLSQ(λs)
res = solve(ddprob, basis, opt, progress = false, sampler = sampler, denoise = false, normalize = false, maxiter = 5000)
println(result(res))
parameters(res)
_prob = ODEProblem(res.basis, u0, tspan, parameters(res))
_sol = solve(_prob, Tsit5())



baddies = Int[]
pls = map(1:length(solutions)) do i
    s = solutions[i]
    _s = solve(_prob, Tsit5(), u0 = s.prob.u0, saveat = 0.01)
    _s_ = solve(prob, Tsit5(), u0 = s.prob.u0, saveat = 0.1, p = s.prob.p)
    _pl = plot(_s, alpha = 0.1, label = nothing)
    plot!(_s_, label = nothing, color = [:blue :red])
    scatter!(s, label = nothing, color = [:blue :red])
    # Individual prediction of the solution
    res_iv = DataDrivenDiffEq.get_solution(res, i)
    __prob = ODEProblem(
        res_iv.basis, s.prob.u0, s.prob.tspan, parameters(res_iv)
    )
    __sol = solve(__prob, Tsit5(), saveat = 0.1, )
    if __sol.retcode != :Success
        push!(baddies, i)
    end
    pl_ = plot(__sol, label = nothing, 
        linestyle = :dash, color = [:blue :red], 
        ylim = (2*minimum(_s_), 2*maximum(_s_)))
    plot!(_s_, label = nothing, color = [:blue :red])
    plot!(_s, label = nothing, alpha = 0.1)
    _pl, pl_
end

baddies

plot(
    plot(last.(pls)[baddies]..., layout = (length(baddies),1)),
    size = (500, 1000)
)

idxs = rand(1:50, 10)

plot(
    plot(last.(pls)[idxs]..., layout = (5,2)),
    size = (1000, 1000)
)

