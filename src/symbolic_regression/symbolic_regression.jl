
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
    eval_expression = false,
    kwargs...
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

    build_solution(prob, alg, hof; eval_expression = eval_expression)
end

function pareto_optimal_equations(hof::HallOfFame, prob, alg)
    return pareto_optimal_equations([hof], prob, alg)
end


function pareto_optimal_equations(hof::Vector{HallOfFame}, prob, alg)

    opts = DataDrivenDiffEq.to_options(alg)
    y = DataDrivenDiffEq.get_target(prob)
    x, _, t, c = DataDrivenDiffEq.get_oop_args(prob)
    X =  vcat([x for x in (x, c, permutedims(t)) if !isempty(x)]...)
    
    @parameters t
    @variables x[1:size(prob.X, 1)](t) u[1:size(prob.U,1)](t)
    x = collect(x)
    u = collect(u)
    x_ = Num[x;u;t]

    # Build a dict
    
    subs = Dict([SymbolicUtils.Sym{LiteralReal}(Symbol("x$(i)")) => x_[i] for i in 1:size(x_, 1)]...)


    eqs = map(1:size(hof, 1)) do i
        d = calculateParetoFrontier(X, y[i,:], hof[i], opts)
        isempty(d) && return Num(0)
        eq_ = node_to_symbolic(last(d).tree, opts)
        substitute(eq_, subs)
    end

    return Num.(eqs), x, u, t
end



function build_solution(prob::AbstractDataDrivenProblem, alg::EQSearch, hof; eval_expression = false)

    eqs, x, u, t = pareto_optimal_equations(hof, prob, alg)
    
    lhs, dt = assert_lhs(prob)


    # Build the lhs
    if (length(eqs) == length(x)) && (lhs != :direct)
            if lhs == :continuous
                d = Differential(t)
            elseif lhs == :discrete
                d = Difference(t, dt = dt)
            end
            eqs = [d(x[i]) ~ eq for (i,eq) in enumerate(eqs)]
    end

    res_ = Basis(
        eqs, x, iv = t, controls = u, eval_expression = eval_expression
    )

    X = get_target(prob)
    Y = res_(get_oop_args(prob)...)



    retcode = :converged 
    
    return DataDrivenSolution(
        res_, [], retcode, alg, hof, prob, false
    )
end
