using DataDrivenDiffEq
using ModelingToolkit
using LinearAlgebra

abstract type AbstractEFSStrategy end

struct FixedSize <: AbstractEFSStrategy end
struct FlexibleSize <: AbstractEFSStrategy end
struct CorrelationFiltering <: AbstractEFSStrategy end

mutable struct EFS{P, S} 
    p::Int64 # initial ops
    q::Int64 # new features from initial ops 
    μ::Int64 # new features from q and p
    opool::P # Operations
    strategy::S # strategy
end

EFS(q, p, μ, ops::OperationPool) = EFS(q, p, μ, ops, FixedSize())

Base.length(e::EFS) = e.p+e.q+e.μ
state_inds(e::EFS) = 1:e.p
feature_inds(e::EFS) = state_inds(e) .+ e.q
composable_inds(e::EFS) = 1:(e.q+e.p)
composed_inds(e::EFS) = composable_inds(e) .+ e.μ 
strategy(e::EFS) = e.strategy

function DataDrivenDiffEq.Candidate(e::EFS, b::Basis)
    @assert length(e) >= length(b) >= e.p
    c = DataDrivenDiffEq.Candidate(b)
    add_features!(c, e)
    return c
end

function add_features!(c::DataDrivenDiffEq.Candidate, e::EFS, s::FixedSize)
    dl = length(e) - length(c)
    dl > 0 ? add_features!(c, e.opool, dl, state_inds(e)) : nothing
    add_features!(c, e.opool, e.q, state_inds(e), feature_inds(e))
    add_features!(c, e.opool, e.μ, composable_inds(e), composed_inds(e))
    return
end

function _symbolic_regression!(Ψ, θ, X, DX, p, t, s::FixedSize ,thresholds, maxiter, opt, denoise, normalize, convergence_error)
    nx, nm = size(X)
    ny, nm = size(DX)

    # Pathwise sparse regression
    scales = ones(eltype(X), length(Ψ))

    ξ_tmp = similar(Ξ)
    Ξ_tmp = zeros(eltype(X), length(Ψ), ny)

    iter_ = Inf
    _iter = Inf
    _thresh = Inf

    denoise ? optimal_shrinkage!(θ') : nothing
    normalize ? normalize_theta!(scales, θ) : nothing
    
    @inbounds for j in 1:length(thresholds)
        set_threshold!(opt, thresholds[j])
        iter_ = sparse_regression!(view(ξ_tmp, :, :), view(θ, :, :), view(DX, :, :), maxiter, opt, false, false, convergence_error)
        normalize ? rescale_xi!(ξ_tmp, scales) : nothing
        for i in 1:ny
            if j == 1
                Ξ_opt .= ξ_tmp
                _iter = iter_
                _thresh = thresholds[j]
            else
                if evaluate_pareto!(view(Ξ_opt, :, i), view(ξ_tmp, :, i), fg, view(θ, :, :), view(DX, i, :))
                    _iter = iter_
                    _thresh = thresholds[j]
                end
            end
        end
    end

    
end

function symbolic_regression(X, DX, b::Basis, thresholds::AbstractArray, e::EFS, opt::T = STRRidge(), steps::Int64 = 10; 
    f::Function = (xi, theta, dx)->[norm(xi, 0); norm(dx .- theta'*xi, 2)], 
    g::Function = x->norm(x), p::AbstractArray = [], t::AbstractVector = [], 
    maxiter::Int64 = 10, denoise::Bool = false, normalize::Bool = true, convergence_error = eps())

    @assert size(X)[end] == size(DX)[end]
    nx, nm = size(X)
    ny, nm = size(DX)

    Ψ = Candidate(b)

    θ = Ψ(X, p, t)

    fg(xi, theta, dx) = (g∘f)(xi, theta, dx)

    for i in 1:steps
        _symbolic_regression!(Ψ, Ξ, X, DX, p, t, strategy(e), thresholds, maxiter, opt, denoise, normalize, convergence_error)
    end
    
    return SparseIdentificationResult(Ξ_opt, Ψ, _iter, opt, _iter < maxiter, DX, X, p = p)
end

@variables x[1:4]
# We generate the operation pool
ops = [sin, cos, tanh, +, *, /, exp]
op = OperationPool(ops)
b = Basis(x, x, simplify_eqs = false)
e = EFS(4, 2, 2, op)
X = randn(4, 10)
A = randn(2,4)
Y = A*X
c = init(e, b, X, Y, STRRidge())
c' \ Y'
A
println(b)
