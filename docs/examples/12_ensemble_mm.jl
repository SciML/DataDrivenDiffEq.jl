using DataDrivenDiffEq
using LinearAlgebra
using ModelingToolkit
using OrdinaryDiffEq
using Plots
using Statistics
using StatsBase

function michaelis_menten(u, p, t)
    [p[1] - p[2]*u[1]/(p[3]+u[1])]
end


u0 = [3.0]
p0 = [0.6; 1.5; 0.3]

u_init() = u0 .* (1 .+ 0.2*(rand()-0.5)) 

p_init() = begin
    p_1 = rand(0.6:0.1:1.4)#p0[1] * (1 .+ 0.2*(rand() - 0.5))
    p_2 = p_1 * rand(1:0.1:2.0)#p0[2] * (1 .+ 0.1*(rand() - 0.5))
    p_3 = p_1 * rand(0.5:0.1:1.0)#p0[3] * (1 + 0.5*(rand() - 0.5))
    vcat(p_1, p_2, p_3)
end

t_observed(tspan) = begin
    δ = -(-(tspan...) / 100)
    sort(vcat(first(tspan), sample(first(tspan):δ:last(tspan), rand(10:1:30), replace = false)))
end


prob = ODEProblem(michaelis_menten, u0, (0.0, 10.0), p0);
sol = solve(prob, Tsit5(), saveat = 0.01)
plot(sol)

solve_realization(prob) = begin
    solve(
        prob, Tsit5(), 
        u0 = u_init(), p = p_init(), saveat  = t_observed(prob.tspan)
    )
end

solutions = map(1:30) do i 
    solve_realization(prob)
end

pl = plot()
map(solutions) do s
    plot!(s, alpha = 0.4, label = nothing)
end
display(pl)

ddprob = DataDrivenEnsemble(solutions...)

# Next, we define our [`Basis`](@ref). Since we want to identify an implicit system, we have to include  
# some candidate terms which use these as an argument and inform our constructor about the meaning of these variables.

@parameters t
D = Differential(t)
@variables u[1:1](t)
h = [monomial_basis(u[1:1], 4)...]
basis = Basis([h; h .* D(u[1])], u, implicits = D.(u), iv = t)
println(basis) # hide
    
sampler = DataSampler(
    Split(ratio = 0.8), Batcher(n = 2)
)

opt = ImplicitOptimizer(1e-1:1e-1:5e-1)
res = solve(ddprob, basis, opt,  normalize = false, denoise = false, by = :min, sampler = sampler, maxiter = 1000);
println(res) # hide
println(result(res))
println(parameters(res))
# As we can see, the [`DataDrivenSolution`](@ref) has good metrics. Furthermore, inspection of the underlying system shows that the original equations have been recovered correctly:
    
_prob = ODEProblem(res.basis, u0, tspan, parameters(res))
_sol = solve(_prob, Tsit5(), saveat = 0.1)
plot(_sol, alpha = 0.8, label = nothing, ylim = (-10,10))
plot!(sol)


baddies = Int[]
pls = map(1:length(res.results)) do i
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
        linestyle = :dash, color = [:red], 
        ylim = (-1.0, 1.5*maximum(_s_)))
    plot!(_s_, label = nothing, color = [:blue])
    plot!(_s, label = nothing, alpha = 0.1)
    _pl, pl_
end

plot(first.(pls)..., size = (1000, 1000))
plot(last.(pls)..., size = (1000, 1000))