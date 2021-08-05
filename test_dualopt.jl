using DataDrivenDiffEq
using ModelingToolkit
using DiffEqBase
using NLsolve
using OrdinaryDiffEq
using LinearAlgebra
using ProgressMeter

function forced_lorenz!(du, u, p, t)
    du[1] = 10(u[2]-u[1]) + 20*tanh(p[1]*t-p[2])
    du[2] = u[1]*(28-u[3])-u[2]
    du[3] = u[1]*u[2]-8/3*u[3]
end

u0 = [8.0; -7.0; 27.0]
tspan = (0.0, 10.0)
dt = 0.005
prob = ODEProblem(forced_lorenz!, u0, tspan, [0.8; 3.0])
sol = OrdinaryDiffEq.solve(prob, Tsit5(), saveat=dt)


@variables u[1:2] p[1:2] t
u = Symbolics.scalarize(u)
p = Symbolics.scalarize(p)

basis = Basis([polynomial_basis(u, 3); tanh(p[1]*t-p[2])], u, parameters = p, independent_variable = t)
println(basis)




## Type
abstract type AbstractAlternatingOptimizer{T} <: DataDrivenDiffEq.Optimize.AbstractOptimizer{T} end;

mutable struct DualSR3{T, V, P <: DataDrivenDiffEq.Optimize.AbstractProximalOperator} <: AbstractAlternatingOptimizer{T}
   """Sparsity threshold"""
   λ::T
   """Relaxation parameter"""
   ν::V
   """Proximal operator"""
   R::P
   """Learning rate"""
   η::T

   function DualSR3(threshold::T = 1e-1, ν::V = 1.0, R::P = HardThreshold(), η::T = 1.0) where {T,V <: Number,P <: DataDrivenDiffEq.Optimize.AbstractProximalOperator}
      @assert all(threshold .> zero(eltype(threshold))) "Threshold must be positive definite"
      @assert ν > zero(V) "Relaxation must be positive definite"

       λ = isa(R, HardThreshold) ? threshold.^2 /2 : threshold
       return new{typeof(λ), V, P}(λ, ν, R, η)
   end

   function DualSR3(threshold::T , R::P, η::T = 1.0) where {T,P <: DataDrivenDiffEq.Optimize.AbstractProximalOperator}
      @assert all(threshold .> zero(eltype(threshold))) "Threshold must be positive definite"
       λ = isa(R, HardThreshold) ? threshold.^2 /2 : threshold
       ν = one(eltype(λ))
       return new{typeof(λ), eltype(λ), P}(λ, ν, R, η)
   end
end

Base.summary(::DualSR3) = "DualSR3"


mutable struct SR3Cache{T, F, S, P}
    "Current coefficient matrix"
    Ξ::AbstractMatrix{T}
    "Previous coefficient matrix"
    Ξ_prev::AbstractMatrix{T}

    "Dictionary"
    Θ::AbstractMatrix{T}
    "Problem"
    prob::DataDrivenProblem
    "Basis"
    Ψ::Basis
    "Jacobian of the basis"
    ∇Ψ::F

    "Non-relaxed coefficient values"
    X::AbstractMatrix{T}
    "Matrices for computation"
    H::AbstractMatrix{T}
    X̂::AbstractMatrix{T}

    "Indictator for normalization"
    normalize::Bool
    "Indictator for denoising"
    denoise::Bool

    "Iterations"
    iters::Int
    maxiters::Int
    "Objective"
    obj::T
    "Sparsity"
    sparsity::T
    "Convergence"
    conv::T

    "Progressbar"
    progress::P
    "Scales for normalization"
    scales::AbstractVector{T}
    "Current parameter values"
    p::AbstractVector{T}
    "Previous parameter values"
    p_prev::AbstractVector{T}
end


function SR3Cache(prob::DataDrivenProblem{dType}, basis, opt, λ = first(opt.λ);
    maxiter = 100, normalize = false, denoise = false, progress = false) where {dType}
    # Generate the matrix
    X, p, t, U = DataDrivenDiffEq.get_oop_args(prob)
    Y = DataDrivenDiffEq.get_target(prob)

    # Evaluate the basis
    θ = basis(DataDrivenDiffEq.get_oop_args(prob)...)

    # Normalize via p norm
    scales = ones(dType, size(θ, 1))

    normalize ? DataDrivenDiffEq.normalize_theta!(scales, θ) : nothing

    # Denoise via optimal shrinkage
    denoise ? DataDrivenDiffEq.optimal_shrinkage!(θ') : nothing

    Ξ = DataDrivenDiffEq.Optimize.init(opt, θ', Y')

    X_ = deepcopy(Ξ)

    # Jacobian w.r.t. parameters
    ∇Ψ = jacobian(basis, parameters(basis))

    H = θ*θ'+I*opt.ν
    X̂ = θ*Y'

    obj = norm(Y - Ξ'θ)
    sparsity = DataDrivenDiffEq.Optimize.norm(Ξ, 0, λ)
    conv = convert(dType, Inf)

    prog = nothing
    if progress
        prog = DataDrivenDiffEq.Optimize.init_progress(
            opt, Ξ, θ', Y', maxiter, 1
        )
    end

    return SR3Cache{dType, typeof(∇Ψ), typeof(obj), typeof(prog)}(
        Ξ, deepcopy(Ξ), θ, prob, basis, ∇Ψ, X_, H, X̂, normalize, denoise, 1, maxiter, obj, sparsity, conv, prog, scales, deepcopy(p), zero(p)
    )
end


function get_oop_args(c::SR3Cache)
    args = DataDrivenDiffEq.get_oop_args(c.prob)
    return (args[1], c.p, args[3:end]...)
end

function get_jacs(c::SR3Cache)
    X, p, t, U = get_oop_args(c)
    if isempty(U)
        return map(i->(c.Ξ)'*c.∇Ψ(X[:,i], p, t[i]), 1:size(X, 2))
    else
        return map(i->(c.Ξ)'*c.∇Ψ(X[:,i], p, t[i], U[:,i]), 1:size(X, 2))
    end
end

function get_Ab(c::SR3Cache)
    j = get_jacs(c)

    A = sum(map(x->x'x, j)) / length(j)

    X, p, t, U = get_oop_args(c)
    Y = DataDrivenDiffEq.get_target(c.prob)
    #idxs = current_idxs(c)
    if isempty(U)
        b =  Y-c.Ξ'*diagm(1 ./c.scales)*c.Ψ(X, p, t)
    else
        b =  Y-c.Ξ'*diagm(1 ./c.scales)*c.Ψ(X, p, t, U)
    end
    b_  = zeros(size(j[1], 2))
    for i in 1:length(j)
        b_ .+= j[i]'b[:,i] / length(j)
    end
    A, b_
end

function update_cache!(c::SR3Cache, opt, λ)
    # Update theta
    c.Ψ(c.Θ, get_oop_args(c)...)
    # Normalize and denoise if necessary
    c.normalize ? DataDrivenDiffEq.normalize_theta!(c.scales, c.Θ) : nothing
    c.denoise ? DataDrivenDiffEq.optimal_shrinkage!(c.Θ') : nothing
    # Update matrices
    c.H .= c.Θ*c.Θ'+I*opt.ν
    c.X̂ .= c.Θ*DataDrivenDiffEq.get_target(c.prob)'
    c.obj = norm(DataDrivenDiffEq.get_target(c.prob) - c.Ξ'*c.Θ)
    c.sparsity = DataDrivenDiffEq.Optimize.norm(c.Ξ, 0, λ)
    c.iters += 1
    c.conv = norm(c.Ξ .- c.Ξ_prev, 2) + norm(c.p .- c.p_prev, 2)
    # History
    c.Ξ_prev .= c.Ξ
    c.p_prev .= c.p

    if !isnothing(c.progress)
        ProgressMeter.next!(
            c.progress;
            showvalues = [
                (:Threshold, λ), (:Objective, c.obj), (:Sparsity, c.sparsity),
                (:Convergence, c.conv), (:Parameters, c.p)
            ]
        )
    end
    return
end

function update_step!(c::SR3Cache, opt::DualSR3, λ::Number = first(opt.λ), args...; kwargs...)

    # Update coefficient
    c.X .= c.H \ (c.X̂ .+ c.Ξ*opt.ν)
    opt.R(c.Ξ, c.X, λ)


    # Just do this if the parameters changed
    if c.iters <  2 || norm(c.p - opt.η*c.p_prev) / norm(c.p) >= 1e-1
        # Update parameters
        A, b = get_Ab(c)

        f!(x, p) = x .= A*p - b
        g!(j, p) = j .= A
        res = nlsolve(f!, g!, zero(c.p), iterations = 1)
        c.p .+= opt.η*res.zero
    end

    # Inplace update of the theta
    update_cache!(c, opt, λ)
end

function update_step!(c::SR3Cache, opt::DualSR3, λ::Number, a::AbstractVector, b::AbstractVector, args...; kwargs...)

    # Update coefficient
    c.X .= c.H \ (c.X̂ .+ c.Ξ*opt.ν)
    opt.R(c.Ξ, c.X, λ)

    # Just do this if the parameters changed
    if c.iters <  2 || norm(c.p - opt.η*c.p_prev) / norm(c.p) >= 1e-1
        # Update parameters
        A, b = get_Ab(c)

        f!(x, p) = x .= A*p - b
        g!(j, p) = j .= A
        res = mcpsolve(f!, g!, zero(c.p),a,b, iterations = 1)
        optcache.p .+= opt.η*res.zero
    end

    # Inplace update of the theta
    update_cache!(c, opt)
end

is_converged(c::SR3Cache, ϵ = eps()) = c.conv < ϵ || c.iters >= c.maxiters

function DiffEqBase.solve(p::DataDrivenProblem{dType}, b::Basis, opt::DualSR3{T, V, P}, lowerbounds = nothing, upperbounds = nothing;
    normalize::Bool = false, denoise::Bool = false, maxiter::Int = 0,
    progress = false,
    round::Bool = true,
    abstol = eps(),
    eval_expression = false, kwargs...) where {dType <: Number, T <: Number, V, P}
    # Check the validity
    @assert is_valid(p) "The problem seems to be ill-defined. Please check the problem definition."

    # Initialize the cache
    cache = SR3Cache(p, b, opt, progress = progress, maxiter = maxiter, normalize = normalize, denoise = denoise)

    # Clamp the paremeters
    if !isnothing(lowerbounds) && !isnothing(upperbounds)
        clamp!.(cache.p, lowerbounds, upperbounds)
    end

    while !is_converged(cache, abstol)
        update_step!(cache, opt, opt.λ, lowerbounds, upperbounds)
    end

    normalize ? DataDrivenDiffEq.rescale_xi!(cache.Ξ, cache.scales, round) : nothing

    # Build solution Basis
    return DataDrivenDiffEq.build_solution(
        p, cache.Ξ, opt, b, eval_expression = eval_expression
    )
end

ddprob = ContinuousDataDrivenProblem(sol, p = [0.35; 2.0])
opt = DualSR3(1e-1, 1.0, SoftThreshold(), 10.0)

res = solve(ddprob, basis, opt, maxiter = 100, progress = true, abstol = 1e-3)
