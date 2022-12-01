function DataDrivenDiffEq.get_fit_targets(::A, prob::AbstractDataDrivenProblem,
                                          basis::Basis) where {
                                                               A <: AbstractDAGSRAlgorithm
                                                               }
    return prob.X, DataDrivenDiffEq.get_implicit_data(prob)
end

function CommonSolve.solve!(prob::InternalDataDrivenProblem{A}) where {
                                                                       A <:
                                                                       AbstractDAGSRAlgorithm
                                                                       }
    @unpack alg, basis, testdata, traindata, control_idx, options, problem, kwargs = prob
    @unpack maxiters, progress, eval_expresssion, abstol = options

    cache = init_cache(alg, basis, problem)

    p = progress ? ProgressMeter.Progress(maxiters, dt = 0.1) : nothing

    for iter in 1:maxiters
        update_cache!(cache)
        if progress
            if StatsBase.rss(first(cache.candidates)) <= abstol
                ProgressMeter.finish!(p)
                break
            end
            ProgressMeter.update!(p, iter, showvalues = [(:Algorithm, cache)])
        end
    end

    # Create the optimal basis
    best_cache = first(sort!(cache.candidates, by = alg.loss))

    p_best = get_parameters(best_cache)

    p_new = map(enumerate(ModelingToolkit.parameters(basis))) do (i, ps)
        DataDrivenDiffEq._set_default_val(Num(ps), p_best[i])
    end

    subs = Dict(a => b for (a, b) in zip(ModelingToolkit.parameters(basis), p_new))

    rhs = map(x -> Num(x.rhs), equations(basis))
    eqs, _ = best_cache.model(rhs, cache.p, best_cache.st)

    eqs = collect(map(eq -> ModelingToolkit.substitute(eq, subs), eqs))

    new_basis = Basis(eqs, states(basis),
                      parameters = p_new, iv = get_iv(basis),
                      controls = controls(basis), observed = observed(basis),
                      implicits = implicit_variables(basis),
                      name = gensym(:Basis),
                      eval_expression = eval_expresssion)
    new_problem = DataDrivenDiffEq.remake_problem(problem, p = p_best)

    rss = sum(abs2,
              new_basis(new_problem) .- DataDrivenDiffEq.get_implicit_data(new_problem))

    return DataDrivenSolution{typeof(rss)}(new_basis, DDReturnCode(1), alg,
                                           AbstractDataDrivenResult[], new_problem,
                                           rss, length(p_new), prob)
end
