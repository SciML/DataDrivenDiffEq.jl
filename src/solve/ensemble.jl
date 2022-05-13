function CommonSolve.solve(prob::DataDrivenEnsemble, args...; kwargs...)
    num_probs = length(prob)
    #results = AbstractDataDrivenProblem[]
    success = zeros(Bool, num_probs)
    results = Vector{AbstractDataDrivenSolution}(undef, num_probs)
    for (i,p) in enumerate(prob.probs)
        try
            results[i] = solve(p, args...; kwargs...)
            success[i] = true
        catch e
            @debug "Failed to solve $(p.name)"
        end
    end
    DataDrivenEnsembleSolution(
        prob, results[success], args...; kwargs...
    )    
end