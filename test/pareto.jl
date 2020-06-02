using DataDrivenDiffEq
using ModelingToolkit
using OrdinaryDiffEq
using LinearAlgebra
using Plots
gr()



# Create a
function pendulum(u, p, t)
    x = u[2]
    y = -9.81sin(u[1]) - 0.1u[2]^3 -0.2*cos(u[1])
    return [x;y]
end

u0 = [0.99π; -1.0]
tspan = (0.0, 20.0)
prob = ODEProblem(pendulum, u0, tspan)
sol = solve(prob, Tsit5(), saveat = 0.3)

plot(sol)

# Create the differential data
DX = similar(sol[:,:])
for (i, xi) in enumerate(eachcol(sol[:,:]))
    DX[:,i] = pendulum(xi, [], 0.0)
end

# Create a basis
@variables u[1:2]
@parameters w[1:2]
# Lots of polynomials
polys = Operation[1]
for i ∈ 1:5
    push!(polys, u.^i...)
    for j ∈ 1:i-1
        push!(polys, u[1]^i*u[2]^j)
    end
end

# And some other stuff
h = [w[1]*cos(u[1]); w[2]*sin(u[1]); u[1]*u[2]; u[1]*sin(u[2]); u[2]*cos(u[2]); polys...]

basis = Basis(h, u, parameters = w)
println(basis)

# Get the reduced basis via the sparse regression
# Thresholded Sequential Least Squares, works fine for more data
# than assumptions, converges fast but fails sometimes with too much noise
opt = STRRidge(1e-2)

res = SparseIdentificationResult[]
λs = exp10.(-5:0.1:-1)


function bla(X, DX, Ψ::Basis, thresholds::AbstractArray ; pf::ParetoFront = ParetoFront(), p::AbstractArray = [], t::AbstractVector = [], maxiter::Int64 = 10, opt::T = Optimize.STRRidge(),denoise::Bool = false, normalize::Bool = true, convergence_error = eps()) where {T <: DataDrivenDiffEq.Optimize.AbstractOptimizer, S <: AbstractSortingMethod}
    @assert size(X)[end] == size(DX)[end]
    nx, nm = size(X)
    ny, nm = size(DX)

    θ = Ψ(X, p, t)

    scales = ones(eltype(X), length(Ψ))

    ξ = zeros(eltype(X), length(Ψ))
    Ξ = zeros(eltype(X), length(thresholds), length(Ψ))
    Ξ_opt = zeros(eltype(X), length(Ψ), ny)

    iters = zeros(Int64, length(thresholds))

    denoise ? optimal_shrinkage!(θ') : nothing
    normalize ? DataDrivenDiffEq.normalize_theta!(scales, θ) : nothing

    _iter = Inf
    _thresh = Inf

    @inbounds for i in 1:ny
        for (j, threshold) in enumerate(thresholds)
            set_threshold!(opt, threshold)
            iters[j] = sparse_regression!(ξ, θ, DX[i, :]', maxiter, opt, false, false, convergence_error)
            normalize ? DataDrivenDiffEq.rescale_xi!(ξ, scales) : nothing
            pc = ParetoCandidate([norm(ξ, 0); norm(DX[i, :] .- θ'*ξ)], ξ)
            add_candidate!(pf, pc)
        end
        idx = sort!(pf)
        Ξ_opt[:, i] .= parameter(best(pf))
        iters[idx] <  _iter ? _iter = iters[idx] : nothing
        thresholds[idx] < _thresh ? _thresh = thresholds[idx] : nothing
        empty!(pf)
    end

    ## Create the evaluation
    #@inbounds for i in 1:ny
    #    pf = ParetoFront(x[:, :, i])
    #    sort!(pf)
    #    idx = best(pf)[1]
    #    iters[idx] <  _iter ? _iter = iters[idx] : nothing
    #    thresholds[idx] < _thresh ? _thresh = thresholds[idx] : nothing
    #    Ξ_opt[:, i] = Ξ[idx, i, :]
    #end

    set_threshold!(opt, _thresh)
    #return Ξ_opt
    return SparseIdentificationResult(Ξ_opt, Ψ, _iter, opt, _iter < maxiter, DX, X, p = p)
end

λs = exp10.(-5:0.1:-1)
# Use SR3 with high relaxation (allows the solution to diverge from LTSQ) and high iterations
opt = STRRidge()
wf = WeigthedExponentialSum()
pf = ParetoFront(sorting = wf)
opt = SR3(1e-3, 5.0)
Ψ = bla(sol[:,1:10], DX[:, 1:10], basis, λs, pf = pf, p = [1.0; 1.0], maxiter = 15000, opt = opt)
print_equations(Ψ)


using BenchmarkTools

Juno.@profiler Ψ = bla(sol[:,:], DX[:, :], basis, λs, pf = pf, p = [1.0; 1.0], maxiter = 10, opt = opt)
@btime Ψ = SInDy($sol[:,:], $DX[:, :], $basis, $λs, p = [1.0; 1.0], maxiter = 100, opt = $opt)

println(Ψ)
print_equations(Ψ)
