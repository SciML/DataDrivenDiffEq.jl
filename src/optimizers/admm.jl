"""
$(TYPEDEF)
`ADMM` is an implementation of Lasso using the alternating direction methods of multipliers and
loosely based on [this implementation](https://web.stanford.edu/~boyd/papers/admm/lasso/lasso.html).
It solves the following problem
```math
\\argmin_{x} \\frac{1}{2} \\| Ax-b\\|_2 + \\lambda \\|x\\|_1
```
# Fields
$(FIELDS)
# Example
```julia
opt = ADMM()
opt = ADMM(1e-1, 2.0)
```
"""
mutable struct ADMM{T, R} <: AbstractOptimizer{T}
    """Sparsity threshold"""
    λ::T
    """Augmented Lagrangian parameter"""
    ρ::R


    function ADMM(threshold::T = 1e-1, ρ::R = 1.0) where {T, R}
        @assert all(threshold .> zero(eltype(threshold))) "Threshold must be positive definite"
        @assert zero(R) < ρ "Augemented lagrangian parameter should be positive definite"
        return new{T, R}(threshold, ρ)
    end
end

Base.summary(::ADMM) = "ADMM"

mutable struct ADMMCache{T, P, B, S, V} <: AbstractOptimizerCache
    X_prev::T
    u::T
    z::T
    A::P
    b::B
    rho::V

    R::S

    X_opt::T
    λ_opt::AbstractVector{V}

    state::OptimizerState{V}
end

@views init_cache(opt::ADMM, X, A, Y, λ = first(opt.λ); kwargs...) = begin
    X_prev = zero(X)
    X_opt = similar(X)
    X_opt .= X
    λ_opt = zeros(typeof(λ), size(X, 2))

    n, m = size(A)
    rho = opt.ρ

    P = factorize(A'A + rho * I)
    b = A'Y

    u = zero(X)
    z = zero(X)
    
    R = SoftThreshold()
    state = OptimizerState(opt; kwargs...)

    return ADMMCache(
        X_prev, u, z, P, b, rho, R, X_opt, λ_opt, state
    )
end


@views function step!(cache::ADMMCache, X, A, Y, λ)
    
    u = cache.u
    z = cache.z
    R = cache.R
    rho = cache.rho
    A_ = cache.A
    b = cache.b

    ldiv!(z, A_, b .+ rho .* (X.-u))
    R(X, z .+ u, λ .* rho)
    u .+= z .- X

    set_cache!(cache, X, A, Y, λ)
    return 
end

