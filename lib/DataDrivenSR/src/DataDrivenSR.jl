module DataDrivenSR

using DataDrivenDiffEq
# Load specific (abstract) types
using DataDrivenDiffEq: AbstractBasis
using DataDrivenDiffEq: AbstractDataDrivenAlgorithm
using DataDrivenDiffEq: AbstractDataDrivenResult
using DataDrivenDiffEq: AbstractDataDrivenProblem
using DataDrivenDiffEq: DDReturnCode, ABSTRACT_CONT_PROB, ABSTRACT_DISCRETE_PROB
using DataDrivenDiffEq: InternalDataDrivenProblem
using DataDrivenDiffEq: is_implicit, is_controlled

using DataDrivenDiffEq.DocStringExtensions
using DataDrivenDiffEq.CommonSolve
using DataDrivenDiffEq.CommonSolve: solve!
using DataDrivenDiffEq.StatsBase
using DataDrivenDiffEq.Parameters

using Reexport

@reexport using SymbolicRegression
"""
$(TYPEDEF)
Options for using SymbolicRegression.jl within the `solve` function.
Automatically creates [`Options`](https://astroautomata.com/SymbolicRegression.jl/stable/api/#Options) with the given specification.
Sorts the operators stored in `functions` into unary and binary operators on conversion.
# Fields
$(FIELDS)
"""
@with_kw struct EQSearch <: AbstractDataDrivenAlgorithm
    "Optionally weight the loss for each y by this value (same shape as y)"
    weights::Union{AbstractMatrix, AbstractVector, Nothing} = nothing
    "The number of processes to use, if you want EquationSearch to set this up automatically."
    numprocs = nothing
    "If you have set up a distributed run manually with procs = addprocs() and @everywhere, pass the procs to this keyword argument."
    procs::Union{Vector{Int}, Nothing} = nothing
    "If using multiprocessing (parallelism=:multithreading), and are not passing procs manually, then they will be allocated dynamically using addprocs. However, you may also pass a custom function to use instead of addprocs. This function should take a single positional argument, which is the number of processes to use, as well as the lazy keyword argument. For example, if set up on a slurm cluster, you could pass addprocs_function = addprocs_slurm, which will set up slurm processes."
    addprocs_function::Union{Function, Nothing} = nothing
    "What parallelism mode to use. The options are :multithreading, :multiprocessing, and :serial. Multithreading uses less memory, but multiprocessing can handle multi-node compute. If using :multithreading mode, the number of threads available to julia are used. If using :multiprocessing, numprocs processes will be created dynamically if procs is unset. If you have already allocated processes, pass them to the procs argument and they will be used. You may also pass a string instead of a symbol."
    parallelism::Union{String, Symbol} = :serial
    "Whether to run (quick) tests before starting the search, to see if there will be any problems during the equation search related to the host environment"
    runtests::Bool = true
    "Options for 'EquationSearch'"
    eq_options::SymbolicRegression.Options = SymbolicRegression.Options()
end

struct SRResult{H, P, T, TE} <: AbstractDataDrivenResult
    halloffame::H
    paretofrontier::P
    testerror::T
    trainerror::TE
    retcode::DDReturnCode
end

is_success(k::SRResult) = getfield(k, :retcode) == DDReturnCode(1)
l2error(k::SRResult) = is_success(k) ? getfield(k, :testerror) : Inf
function l2error(k::SRResult{<:Any, <:Any, <:Any, Nothing})
    is_success(k) ? getfield(k, :traineerror) : Inf
end

# apply the algorithm on each dataset
function (x::EQSearch)(ps::InternalDataDrivenProblem{EQSearch}, X, Y)
    @unpack problem, testdata, options = ps
    @unpack maxiters, abstol = options
    @unpack weights, eq_options, numprocs, procs, parallelism, runtests = x

    hofs = SymbolicRegression.EquationSearch(X, Y,
                                             niterations = maxiters,
                                             weights = weights,
                                             options = eq_options,
                                             numprocs = numprocs,
                                             procs = procs, parallelism = parallelism,
                                             runtests = runtests)
    
    # We always want something which is a vector or tuple
    hofs = !isa(hofs, AbstractVector) ? [hofs] : hofs
    
    # Evaluate over the full training data
    paretos = map(enumerate(hofs)) do (i, hof)
        SymbolicRegression.calculate_pareto_frontier(X, Y[i, :], hof, eq_options)
    end

    # Trainingerror
    trainerror = mean(x -> x[end].loss, paretos)
    # Testerror 
    X̃, Ỹ = testdata
    if !isempty(X̃)
        testerror = mean(map(enumerate(hofs)) do (i, hof)
                             doms = SymbolicRegression.calculate_pareto_frontier(X̃,
                                                                                 Ỹ[i, :],
                                                                                 hof,
                                                                                 eq_options)
                             doms[end].loss
                         end)
        retcode = testerror <= abstol ? DDReturnCode(1) : DDReturnCode(5)
    else
        testerror = nothing
        retcode = trainerror <= abstol ? DDReturnCode(1) : DDReturnCode(5)
    end

    return SRResult(hofs, paretos, testerror, trainerror, retcode)
end

function convert_to_basis(res::SRResult, prob)
    @unpack paretofrontier = res
    @unpack alg, basis, problem, options = prob
    @unpack eq_options = alg
    @unpack maxiters, eval_expresssion, generate_symbolic_parameters, digits, roundingmode = options

    eqs_ = Num.(map(paretofrontier) do dom
                    node_to_symbolic(dom[end].tree, eq_options)
                end)

    # Substitute with the basis elements
    atoms = map(xi -> xi.rhs, equations(basis))

    subs = Dict([SymbolicUtils.Sym{LiteralReal}(Symbol("x$(i)")) => x
                 for (i, x) in enumerate(atoms)]...)
    eqs = map(Base.Fix2(substitute, subs), eqs_)

    # Get the lhs
    causality, dt = DataDrivenDiffEq.assert_lhs(problem)

    xs = states(basis)

    # Else just keep equations, since its a direct problem
    if causality == :continuous
        d = Differential(get_iv(basis))
        eqs = [d(xs[i]) ~ eq for (i, eq) in enumerate(eqs)]
    elseif causality == :discrete
        d = Difference(get_iv(basis), dt = dt)
        eqs = [d(xs[i]) ~ eq for (i, eq) in enumerate(eqs)]
    else
        phi = [Symbolics.variable(Symbol("φ"), i) for i in 1:length(eqs)]
        eqs = [phi[i] ~ eq for (i, eq) in enumerate(eqs)]
    end

    ps = parameters(basis)
    @unpack p = problem

    p_new = map(eachindex(p)) do i
        DataDrivenDiffEq._set_default_val(Num(ps[i]), p[i])
    end

    Basis(eqs, states(basis),
          parameters = p_new, iv = get_iv(basis),
          controls = controls(basis), observed = observed(basis),
          implicits = implicit_variables(basis),
          name = gensym(:Basis),
          eval_expression = eval_expresssion)
end

function CommonSolve.solve!(ps::InternalDataDrivenProblem{EQSearch})
    @unpack alg, basis, testdata, traindata, kwargs = ps
    @unpack weights, numprocs, procs, addprocs_function, parallelism, runtests, eq_options = alg
    @unpack traindata, testdata, basis, options = ps
    @unpack maxiters, eval_expresssion, generate_symbolic_parameters, digits, roundingmode = options
    @unpack problem = ps

    results = map(traindata) do (X, Y)
        alg(ps, X, Y)
    end

    # Get the best result based on test error, if applicable else use testerror
    sort!(results, by = l2error)
    # Convert to basis
    best_res = first(results)

    new_basis = convert_to_basis(best_res, ps)

    DataDrivenSolution(new_basis, problem, alg, results, ps, best_res.retcode)
end

export EQSearch

end # module
