function CommonSolve.solve!(ps::InternalDataDrivenProblem{<:AbstractSparseRegressionAlgorithm})
    @unpack alg, basis, testdata, traindata, problem, options, transform = ps

    results = map(traindata) do (X, Y)
        __sparse_regression(ps, X, Y)
    end

    # Get the best result based on test error, if applicable else use testerror
    sort!(results, by = l2error)

    # Convert to basis
    best_res = first(results)

    # Transform the best coefficients
    coefficients = permutedims(copy(get_coefficients(best_res)))
    coefficients = permutedims(StatsBase.transform(transform, coefficients))
    new_basis = DataDrivenDiffEq.__construct_basis(coefficients, basis, problem, options)

    DataDrivenSolution(new_basis, problem, alg, results, ps, best_res.retcode)
end

function __sparse_regression(ps::InternalDataDrivenProblem{<:AbstractSparseRegressionAlgorithm}, X::AbstractArray, Y::AbstractArray)
    @unpack alg, testdata, options, transform = ps
    
    coefficients, optimal_thresholds, optimal_iterations = alg(X, Y, options = options)
    
    trainerror = sum(abs2, Y .- coefficients*X)
    
    X̃, Ỹ = testdata
    
    if !isempty(X̃)
        testerror = sum(abs2, Ỹ .- coefficients*X̃)
    else
        testerror = nothing
    end

    retcode = DDReturnCode(1)

    dof = sum(abs.(coefficients) .> 0.)

    SparseRegressionResult(
        coefficients, dof, optimal_thresholds, 
        optimal_iterations, testerror, trainerror, 
        retcode
    )
end


function __sparse_regression(ps::InternalDataDrivenProblem{<:ImplicitOptimizer}, X::AbstractArray, Y::AbstractArray)
    @unpack alg, testdata, options, transform, basis, problem, implicit_idx = ps
    @assert DataDrivenDiffEq.is_implicit(basis) "The provided `Basis` does not have implicit variables!"

    candidate_matrix = zeros(Bool, length(basis), length(DataDrivenDiffEq.implicit_variables(basis)))
    idx = ones(Bool, size(candidate_matrix, 2))
    
    for i in axes(candidate_matrix, 1), j in axes(candidate_matrix, 2)
        idx .= true
        idx[j] = false
        candidate_matrix[i,j] = sum(implicit_idx[i, idx]) == 0
    end

    opt_coefficients = zeros(eltype(problem), size(candidate_matrix, 2), size(candidate_matrix, 1))
    opt_thresholds = []
    opt_iterations = []

    foreach(enumerate(eachcol(candidate_matrix))) do (i,idx)
        coeff, thresholds, iters = alg(X[idx, :], Y, options = options)
        opt_coefficients[i:i,idx] .= coeff
        push!(opt_thresholds, thresholds)
        push!(opt_iterations, iters)
    end
    
    trainerror = sum(abs2, opt_coefficients*X)
    
    X̃, Ỹ = testdata
    
    if !isempty(X̃)
        testerror = sum(abs2, opt_coefficients*X̃)
    else
        testerror = nothing
    end

    retcode = DDReturnCode(1)

    dof = sum(abs.(opt_coefficients) .> 0.)

    SparseRegressionResult(
        opt_coefficients, dof, opt_thresholds, 
        opt_iterations, testerror, trainerror, 
        retcode
    )
end
