function DataDrivenDiffEq.get_fit_targets(::A, prob::AbstractDataDrivenProblem,
                                          basis::Basis) where {
                                                               A <: AbstractDAGSRAlgorithm
                                                               }
    return prob.X, DataDrivenDiffEq.get_implicit_data(prob)
end

struct DataDrivenLuxResult <: DataDrivenDiffEq.AbstractDataDrivenResult
    candidate::Candidate
    retcode::DDReturnCode
end

function CommonSolve.solve!(prob::InternalDataDrivenProblem{A}) where {
                                                                       A <:
                                                                       AbstractDAGSRAlgorithm
                                                                       }
    @unpack alg, basis, testdata, traindata, control_idx, options, problem, kwargs = prob
    @unpack maxiters, progress, eval_expresssion, abstol = options

    cache = init_cache(alg, basis, problem)

    p = ProgressMeter.Progress(maxiters, dt = 0.1, enabled = progress)

    _showvalues = let cache = cache
        (iter) -> begin 
        shows = min(5, sum(cache.keeps))
        losses = map(alg.loss, cache.candidates[cache.keeps])
        min_, max_ = extrema(losses)
        [
            (:Iterations, iter),
            (:RSS, map(StatsBase.rss,cache.candidates[cache.keeps][1:shows])),
            (:Minimum, min_),
            (:Maximum, max_),
            (:Mode, mode(losses)),
            (:Mean, mean(losses)),
            (:Probabilities, map(x->exp(x(cache.p)), cache.candidates[cache.keeps][1:shows]))
        ]
        end
    end

    for iter in 1:maxiters
        update_cache!(cache)
        
        if StatsBase.rss(first(cache.candidates)) <= abstol
            ProgressMeter.finish!(p)
            break
        end
        ProgressMeter.update!(p, iter, showvalues = _showvalues(iter))
    end

    # Create the optimal basis
    sort!(cache.candidates, by = alg.loss)
    best_cache = first(cache.candidates)

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
                                           [cache], new_problem,
                                           rss, length(p_new), prob)
end



