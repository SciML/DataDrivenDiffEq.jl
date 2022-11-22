function DataDrivenDiffEq.get_fit_targets(::A, prob::AbstractDataDrivenProblem,
    basis::Basis) where {
                                 A <: AbstractDAGSRAlgorithm
                                 }

    return prob.X , DataDrivenDiffEq.get_implicit_data(prob)
end

function CommonSolve.solve!(prob::InternalDataDrivenProblem{A}) where {A <: AbstractDAGSRAlgorithm}
    @unpack alg, basis, testdata, traindata, control_idx, options, problem, kwargs = prob
    @unpack maxiters, progress, eval_expresssion = options

    # We do not use the normalized data here 
    # since our Basis contains parameters
    X, _, t, U = DataDrivenDiffEq.get_oop_args(problem)
    Y = DataDrivenDiffEq.get_implicit_data(problem)

    cache = SearchCache(alg, basis, X, Y, U, t)
    p = progress ? ProgressMeter.Progress(maxiters, dt=1.0) : nothing

    foreach(1:maxiters) do iter
        cache = update_cache!(cache)
        if progress
            ProgressMeter.update!(p, iter)
        end
    end

    # Create the optimal basis
    best_cache = first(cache.candidates)
    p_best = get_parameters(best_cache)
    p_new = map(enumerate(ModelingToolkit.parameters(basis))) do (i, ps)
        DataDrivenDiffEq._set_default_val(Num(ps), p_best[i])
    end
    subs = Dict(a => b for (a, b) in zip(ModelingToolkit.parameters(basis), p_new))

    rhs = map(x->Num(x.rhs), equations(basis))
    rhs = collect(map(Base.Fix2(ModelingToolkit.substitute, subs), rhs))
    eqs, _ = cache.model(rhs, cache.p, best_cache.st)
    

    new_basis = Basis(eqs, states(basis),
        parameters = p_new, iv = get_iv(basis),
        controls = controls(basis), observed = observed(basis),
        implicits = implicit_variables(basis),
        name = gensym(:Basis),
        eval_expression = eval_expresssion
    )
    new_problem = DataDrivenDiffEq.remake_problem(problem, p = p_best)
    
    return DataDrivenSolution{typeof(rss(best_cache))}(
        new_basis, DDReturnCode(1), alg, AbstractDataDrivenResult[], new_problem, 
        rss(best_cache), length(p_new), prob
    )
end
