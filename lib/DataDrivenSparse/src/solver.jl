struct SparseLinearSolver{A <: AbstractSparseRegressionAlgorithm, T <: Number}
    algorithm::A
    abstol::T
    reltol::T
    maxiters::Int
    verbose::Bool
    progress::Bool
end

function SparseLinearSolver(x::A; options = DataDrivenCommonOptions()) where A <: AbstractSparseRegressionAlgorithm
    return SparseLinearSolver(
        x, 
        options.abstol, options.reltol, options.maxiters, 
        options.verbose, options.progress
    )
end

init_cache(alg::SparseLinearSolver, X, Y) = init_cache(alg.algorithm, X, Y)

function (alg::SparseLinearSolver)(X::AbstractMatrix, Y::AbstractMatrix)
    @unpack verbose = alg
    map(axes(Y, 1)) do i 
        if verbose
            if i > 1 
                @printf "\n"
            end
            @printf "Starting regression on target variable %6d\n" i  
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

    if verbose
        @printf "Threshold     Iter   DOF   RSS           AICC\n" 
    end
    
    for λ in thresholds

        for iter in 1:maxiters
        
            if iter > 1
                _is_converged(cache, abstol, reltol) && break
            end

            step!(cache, λ)

            if aicc(cache) <= aicc(best_cache)
                _set!(best_cache, cache)
                new_best = true
            end

            if verbose
                show((best_cache, iter, λ))    
            end
        end
    end

    return best_cache
end
