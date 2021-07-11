using SymbolicRegression
using DataDrivenDiffEq


"""
$(TYPEDEF)

Options for using SymbolicRegression.jl within the `solve` function.
Automatically creates [`Options`](https://astroautomata.com/SymbolicRegression.jl/stable/api/#Options) with the given specification.
Sorts the operators stored in `functions` into unary and binary operators on conversion.

# Fields
$(FIELDS)

"""
struct EQSearch <: AbstractSymbolicRegression
    "Operators used for symbolic regression"
    functions::AbstractVector{Function}
    "Additionally keyworded arguments passed to SymbolicRegression.Options"
    kwargs
end

function EQSearch(functions = Function[/, +, *, exp, cos]; kwargs...)
    return EQSearch(functions, kwargs)
end

function to_options(x::EQSearch)
    binops = Tuple([fi for fi in x.functions if is_binary(fi)])
    unaops = Tuple([fi for fi in x.functions if is_unary(fi)])
    fnames = fieldnames(SymbolicRegression.Options)
    ks = Dict(
        [k => v for (k,v) in x.kwargs if k ∈ fnames]
    )
    return SymbolicRegression.Options(;
        binary_operators = binops, unary_operators = unaops,
        ks...
    )
end

function DiffEqBase.solve(prob::AbstractDataDrivenProblem, alg::EQSearch;
    max_iter::Int = 10,
    weights = nothing,
    numprocs = nothing, procs = nothing,
    multithreading = false,
    runtests::Bool = true,
    eval_expression = false
    )

    opt = to_options(alg)

    Y = get_target(prob)

    # Inputs
    X̂, _, t, U = get_oop_args(prob)

    t = iszero(t) ? [] : t
    # Cat the inputs
    X = vcat([x for x in (X̂, U, permutedims(t)) if !isempty(x)]...)
    # Use the eqssearch of symbolic regression
    hof = SymbolicRegression.EquationSearch(X, Y, niterations = max_iter, weights = weights, options = opt,
            numprocs = numprocs, procs = procs, multithreading = multithreading,
            runtests = runtests)
    # Sort the paretofront
    doms = map(1:size(Y, 1)) do i
        calculateParetoFrontier(X, Y[i, :], hof[i], opt)
    end

    build_solution(prob, alg, doms; eval_expression = eval_expression)
end


function build_solution(prob::AbstractDataDrivenProblem, alg::EQSearch, doms; eval_expression = false)

    opt = to_options(alg)
    @variables x[1:size(prob.X, 1)] u[1:size(prob.U,1)] t
    x = Symbolics.scalarize(x)
    u = Symbolics.scalarize(u)
    x_ = [x;u;t]

    # Build a dict
    subs = Dict([SymbolicUtils.Sym{Number}(Symbol("x$(i)")) => x_[i] for i in 1:size(x_, 1)]...)
    # Create a variable
    eqs = vcat(map(x->node_to_symbolic(x[end].tree, opt), doms))
    eqs = map(x->substitute(x, subs), eqs)
    res_ = Basis(
        eqs, x, iv = t, controls = u, eval_expression = eval_expression
    )

    X = get_target(prob)
    Y = res_(get_oop_args(prob)...)

    # Build the metrics
    complexities = map(x->countNodes(x[end].tree), doms)
    complexity = sum(complexities)
    retcode = :sucess

    error = norm(X-Y, 2)
    k = free_parameters(res_)
    aic = AICC(k, X, Y)
    errors = zeros(eltype(X), size(Y, 1))
    aiccs = zeros(eltype(X), size(Y, 1))
    j = 1
    for i in 1:size(Y,1)
        errors[i] = norm(X[i,:].-Y[i,:],2)
        aiccs[i] = AICC(k, X[i:i, :], Y[i:i,:])
    end

    metrics = (
        Complexity = complexity,
        Error = error,
        AICC = aic,
        complexities = complexities,
        Errors = errors,
        AICCs = aiccs,
    )

    inputs = (
        Problem = prob,
        Algorithm = opt,
    )

    return DataDrivenSolution(
        res_, retcode, [], alg, doms, inputs, metrics
    )
end
