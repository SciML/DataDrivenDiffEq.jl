"""
$(TYPEDEF)

Sparse regression solver that applies an [`AbstractSparseRegressionAlgorithm`](@ref)
to one or more target variables.

# Arguments

- `algorithm::AbstractSparseRegressionAlgorithm`: algorithm used for each target.

# Keywords

- `options::DataDrivenCommonOptions`: tolerances, iteration limits, selector, and
  progress settings copied into the solver.

# Returns

Return a solver object callable as `solver(X, Y)`, where `X` has features in rows
and `Y` has target variables in rows. The call returns one cache, selected
threshold, and iteration count per target.

## Fields

$(TYPEDFIELDS)
"""
struct SparseLinearSolver{A <: AbstractSparseRegressionAlgorithm, T <: Number}
    """Sparse-regression algorithm applied to each target."""
    algorithm::A
    """Absolute convergence tolerance for cache updates."""
    abstol::T
    """Relative convergence tolerance for cache updates."""
    reltol::T
    """Maximum number of iterations over all thresholds."""
    maxiters::Int
    """Whether progress information is printed."""
    verbose::Bool
    """Whether the underlying algorithm reports progress."""
    progress::Bool
    """Function used to select the best cache."""
    selector::Function
end

function SparseLinearSolver(
        x::A;
        options = DataDrivenCommonOptions()
    ) where {
        A <:
        AbstractSparseRegressionAlgorithm,
    }
    return SparseLinearSolver(
        x,
        options.abstol, options.reltol, options.maxiters,
        options.verbose, options.progress, options.selector
    )
end

init_cache(alg::SparseLinearSolver, X, Y) = init_cache(alg.algorithm, X, Y)

function (alg::SparseLinearSolver)(X::AbstractMatrix, Y::AbstractMatrix)
    @unpack verbose = alg
    return map(axes(Y, 1)) do i
        if verbose
            if i > 1
                @printf "\n"
            end
            @printf "Starting sparse regression on target variable %6d\n" i
        end
        alg(X, Y[i, :])
    end
end

function (alg::SparseLinearSolver)(X::AbstractArray, Y::AbstractVector)
    @unpack algorithm, abstol, reltol, maxiters, verbose, progress = alg

    thresholds = get_thresholds(algorithm)

    if !issorted(thresholds)
        sort!(thresholds)
    end

    cache = init_cache(alg, X, Y)
    best_cache = init_cache(alg, X, Y)
    _zero!(best_cache)
    new_best = false

    optimal_threshold = minimum(thresholds)
    optimal_iterations = 0
    iteration_counter = 0

    if verbose
        @printf "Threshold     Iter   DOF   RSS           AICC          Updated result\n"
    end

    for (j, λ) in enumerate(thresholds)
        for iter in 1:maxiters
            iteration_counter += 1

            step!(cache, λ)

            if (alg.selector(cache) <= alg.selector(best_cache)) || (j == 1)
                _set!(best_cache, cache)
                new_best = true
                optimal_iterations = iteration_counter
                optimal_threshold = λ
            else
                new_best = false
            end

            if verbose
                @printf "%14e %6d %6d   %14e   %14e %1d\n" λ iter dof(best_cache) rss(best_cache) aicc(best_cache) new_best
            end

            _is_converged(cache, abstol, reltol) && break
        end
    end

    return best_cache, optimal_threshold, iteration_counter
end
