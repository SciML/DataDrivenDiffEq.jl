
"""
$(SIGNATURES)

Implements a sparse regression, given an `AbstractOptimizer`.
"""
function sparse_regression!(X::AbstractArray, A::AbstractArray, Y::AbstractArray, opt::AbstractOptimizer{T};
    maxiter::Int = maximum(size(A)),
    abstol = eps(eltype(T)), progress::Bool = false) where T <: Number

    λ = first(opt.λ)

    if progress
        progress =  Progress(maxiter*length(λ), 1, "Solving sparse regression...")
    else
        progress = nothing
    end

    @views opt(X, A, Y, λ, maxiter = maxiter, abstol = abstol, progress = progress)

    return
end

function sparse_regression!(X::AbstractArray, A::AbstractArray, Y::AbstractArray, opt::AbstractOptimizer{T};
    maxiter::Int = maximum(size(A)),
    abstol = eps(eltype(T)), progress::Bool = false,
    f::Function = F(opt),
    g::Function = G(opt)) where T <: AbstractVector

    # Closure for the pareto function
    fg(x, A, y) = (g∘f)(x, A, y)
    #return fg

    # TODO Tmp Result
    X_tmp = deepcopy(X)
    X_opt = deepcopy(X)

    λ = opt.λ

    if progress
        progress =  Progress(maxiter*length(λ), 1, "Solving sparse regression...")
    else
        progress = nothing
    end

    @views for (i,λi) in enumerate(λ)
        opt(X_tmp, A, Y, λi, maxiter = maxiter, abstol = abstol, progress = progress)
        for j in 1:size(Y, 2)
            evaluate_pareto!(view(X_opt, :, j), view(X_tmp, :, j), fg, view(A, :, :), view(Y, :, j))
        end
    end

    X .= X_opt
    return
end
