# Simple ridge regression based upon the sindy-mpc
# repository, see https://arxiv.org/abs/1711.05501
# and https://github.com/eurika-kaiser/SINDY-MPC/blob/master/LICENSE

"""
$(TYPEDEF)
`STLQS` is taken from the [original paper on SINDY](https://www.pnas.org/content/113/15/3932) and implements a
sequentially thresholded least squares iteration. `λ` is the threshold of the iteration.
It is based upon [this matlab implementation](https://github.com/eurika-kaiser/SINDY-MPC/utils/sparsifyDynamics.m).
It solves the following problem
```math
\\argmin_{x} \\frac{1}{2} \\| Ax-b\\|_2 + \\lambda \\|x\\|_2
```

# Fields
$(FIELDS)

# Example
```julia
opt = STLQS()
opt = STLQS(1e-1)
opt = STLQS(Float32[1e-2; 1e-1])
```

## Note
This was formally `STRRidge` and has been renamed.
"""
mutable struct STLSQ{T} <: AbstractOptimizer{T}
    """Sparsity threshold"""
    λ::T

    function STLSQ(threshold::T = 1e-1) where T
        @assert all(threshold .> zero(eltype(threshold))) "Threshold must be positive definite"

        return new{typeof(threshold)}(threshold)
    end

end

Base.summary(::STLSQ) = "STLSQ"

mutable struct STLSQCache{T, B, V} <: AbstractOptimizerCache
    X_prev::T
    biginds::B

    X_opt::T
    λ_opt::AbstractVector{V}

    state::OptimizerState{V}
end

@views init_cache(opt::STLSQ, X, A, Y, λ = first(opt.λ); kwargs...) = begin
    X_prev = zero(X)
    X_opt = similar(X)
    X_opt .= zero(X)
    λ_opt = zeros(typeof(λ), size(X, 2))
    biginds = abs.(X) .> λ
    state = OptimizerState(opt; kwargs...)
    return STLSQCache(X_prev, biginds, X_opt, λ_opt,  state) 
end

@views set_cache!(s::STLSQCache, X, A, Y, λ) = begin
    is_convergend!(s.state, X, s.X_prev) && return
    s.biginds .= abs.(X) .> λ
    s.X_prev .= X
    set_metrics!(s.state, A, X, Y, λ)
    eval_pareto!(s, s.state, A, Y, λ)
    increment!(s.state)
    print(s.state, λ)
    return
end

@views function step!(cache::STLSQCache, X, A, Y, λ)
    biginds = cache.biginds
    X .= zero(X)
    for i in axes(Y, 2)
        X[biginds[:, i], i] .= A[:, biginds[:, i]] \ Y[:, i]
    end 
    set_cache!(cache, X, A, Y, λ)
end

