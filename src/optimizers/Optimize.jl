
# Overload the norm
LinearAlgebra.norm(x, p, λ) = norm(x[abs.(x) .> λ], p)

# Dispatch on vector
function (opt::AbstractOptimizer{T} where T)(
        X::AbstractArray, A::AbstractArray, Y::AbstractVector{V} where V,
        args...; kwargs...
    )
    Y = reshape(Y, 1, length(Y))
    return opt(X, A, Y, args...; kwargs...)
end

function (opt::AbstractOptimizer{T} where T)(
        X::AbstractArray, A::AbstractArray, Y::Adjoint{V, AbstractVector{V}} where V,
        args...; kwargs...
    )
    Y = reshape(Y, 1, length(Y))
    return opt(X, A, Y, args...; kwargs...)
end

@views function optimize!(cache::AbstractOptimizerCache, X, A, Y, λ)
    while is_runable(cache)
        step!(cache, X, A, Y, λ)
        # If all coefficients go to zero, break
        all(X .≈ zero(λ)) && break
    end
end

@views function (opt::AbstractOptimizer)(X, A, Y; kwargs...)
    
    cache = init_cache(opt, X, A, Y, first(opt.λ); kwargs...)

    for λ in opt.λ
        optimize!(cache, X, A, Y, λ) 
        reset!(cache)
    end

    copyto!(X ,cache.X_opt)
    return cache.λ_opt
end

reset!(s::AbstractOptimizerCache) = reset!(s.state)

is_runable(s::AbstractOptimizerCache) = is_runable(s.state) 

@views set_cache!(s::AbstractOptimizerCache, X, A, Y, λ) = begin
    is_convergend!(s.state, X, s.X_prev) && return
    copyto!(s.X_prev, X)
    set_metrics!(s.state, A, X, Y, λ)
    eval_pareto!(s, s.state, A, Y, λ)
    increment!(s.state)
    print(s.state, λ)
    return
end


"""
$(SIGNATURES)

Set the threshold(s) of an optimizer to (a) specific value(s).
"""
function set_threshold!(opt::AbstractOptimizer{T}, threshold::T) where T 
    @assert all(threshold .> zero(T)) "Threshold must be positive definite"
    opt.λ .= threshold
end


"""
$(SIGNATURES)

Get the threshold(s) of an optimizer.
"""
get_threshold(opt::AbstractOptimizer) = opt.λ

"""
$(SIGNATURES)

Initialize the optimizer with the least square solution for explicit and `zeros` for implicit optimization.
"""
CommonSolve.init(o::AbstractOptimizer, A, Y) = A \ Y

CommonSolve.init(o::AbstractSubspaceOptimizer, A, Y) = zeros(eltype(A), size(A, 2), size(Y, 2))

"""
$(SIGNATURES)

Initialize the optimizer with the least square solution for explicit and `zeros` for implicit optimization in place.
"""
init!(X::AbstractArray, o::AbstractOptimizer, A::AbstractArray, Y::AbstractArray) =  begin
    @static if VERSION < v"1.7.0"
        ldiv!(X, qr(A, Val(true)), Y)
    else
        ldiv!(X, qr(A, ColumnNorm()), Y)
    end
end

include("./utils.jl")

include("./proximals.jl")

include("./state.jl")

# Remove the trace right now

include("./stlsq.jl")
include("./admm.jl")
include("./sr3.jl")

#Nullspace for implicit sindy
include("./implicit.jl")


# Init the progressmeters
# For a general optimizer



include("./sparseregression.jl")
