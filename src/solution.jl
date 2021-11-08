"""
$(TYPEDEF)

The solution to a `DataDrivenProblem` derived via a certain algorithm.
The solution is represented via an `AbstractBasis`, which makes it callable.

# Fields
$(FIELDS)
"""
struct DataDrivenSolution{R <: Union{AbstractBasis,Nothing} , S, P , A, O,M} <: AbstractDataDrivenSolution
    """Result"""
    res::R # The result
    """Returncode"""
    retcode::Symbol
    """Parameters used to derive the equations"""
    ps::S
    """Algorithm used for solution"""
    alg::A # Solution algorithm
    """Original Output"""
    out::O # E.g. coefficients
    """Original Input"""
    inp::P
    """Error metrics"""
    metrics::M # Named tuples
end

# Make it callable
(r::DataDrivenSolution)(args...) = r.res(args...)

is_implicit(r::DataDrivenSolution) = isa(r.alg, Optimize.AbstractSubspaceOptimizer)

"""
$(SIGNATURES)

Returns the result of in form of an `AbstractBasis`.
"""
result(r::DataDrivenSolution) = r.res

"""
$(SIGNATURES)

Returns the estimated parameters in form of an `Vector`.
"""
ModelingToolkit.parameters(r::DataDrivenSolution) = r.ps

"""
$(SIGNATURES)

Generate a mapping of the parameter values and symbolic representation useable
to `solve` and `ODESystem`.
"""
function parameter_map(r::DataDrivenSolution)
    return [
        ps_ => p_ for (ps_, p_) in zip(parameters(r.res), r.ps)
    ]
end

"""
$(SIGNATURES)

Returns the metrics of the result in form of a `NamedTuple`.
"""
metrics(r::DataDrivenSolution) = r.metrics

"""
$(SIGNATURES)

Returns the original output of the algorithm, e.g. an `AbstractArray` of coefficients for sparse regression.
"""
output(r::DataDrivenSolution) = r.out

"""
$(SIGNATURES)

Returns the algorithm used to derive the solution.
"""
algorithm(r::DataDrivenSolution) = r.alg

"""
$(SIGNATURES)

Returns the original inputs, most commonly the `DataDrivenProblem` and the `Basis` used to derive the solution.
"""
inputs(r::DataDrivenSolution) = r.inp


function Base.summary(io::IO, r::DataDrivenSolution)
    is_implicit(r) ? println(io,"Implicit Result") : println(io,"Explicit Result")
    println(io, "Solution with $(length(r.res)) equations and $(length(r.ps)) parameters.")
    println(io, "Returncode: $(r.retcode)")
    haskey(r.metrics, :Sparsity) && println(io, "Sparsity: $(r.metrics.Sparsity)")
    haskey(r.metrics, :Error) && println(io, "L2 Norm Error: $(r.metrics.Error)")
    haskey(r.metrics, :AICC) && println(io, "AICC: $(r.metrics.AICC)")
    return
end


function Base.print(io::IO, r::DataDrivenSolution, fullview::DataType)

    fullview != Val{true} && return summary(io, r)

    is_implicit(r) ? println(io,"Implicit Result") : println(io,"Explicit Result")
    println(io, "Solution with $(length(r.res)) equations and $(length(r.ps)) parameters.")
    println(io, "Returncode: $(r.retcode)")
    haskey(r.metrics, :Sparsity) && println(io, "Sparsity: $(r.metrics.Sparsity)")
    haskey(r.metrics, :Error) && println(io, "L2 Norm Error: $(r.metrics.Error)")
    haskey(r.metrics, :AICC) && println(io, "AICC: $(r.metrics.AICC)")
    println(io, "")
    print(io, r.res)
    println(io, "")
    if length(r.res.ps) > 0
        x = parameter_map(r)
        println(io, "Parameters:")
        for v in x
            println(io, "   $(v[1]) : $(v[2])")
        end
    end

    return
end

Base.print(io::IO, r::DataDrivenSolution) = summary(io, r)
Base.show(io::IO, r::DataDrivenSolution) = is_implicit(r) ? show(io,"Implicit Result") : show(io,"Explicit Result")


function build_parametrized_eqs(X::AbstractMatrix, b::Basis)
    # Create additional variables
    sp = Int(norm(X, 0))
    sps = norm.(eachcol(X), 0)
    inds = sps .> zero(eltype(X))
    pl = length(parameters(b))
    
    p = [Symbolics.variable(:p, i) for i in (pl+1):(pl+sp)]
    p = collect(p)
    ps = zeros(eltype(X), sp)

    eqs = zeros(Num, sum(inds))
    eqs_ = [e.rhs for e in equations(b)]
    cnt = 1
    for j in 1:size(X, 2)
        if sps[j] == zero(eltype(X))
            continue
        end
        for i in 1:size(X, 1)
            if iszero(X[i,j])
                continue
            end
            ps[cnt] = X[i,j]
            eqs[j] += p[cnt]*eqs_[i]
            cnt += 1
        end
    end
    return eqs, ps, p
end



# Explicit sindy
function build_solution(prob::DataDrivenProblem, Ξ::AbstractMatrix, opt::Optimize.AbstractOptimizer, b::Basis;
    eval_expression = false)
    if all(iszero.(Ξ))
        @warn "Sparse regression failed! All coefficients are zero."
        return DataDrivenSolution(
        nothing , :failed, nothing, opt, Ξ, (Problem = prob, Basis = b, nothing),
    )
    end

    eqs, ps, p_ = build_parametrized_eqs(Ξ, b)

    # Build the lhs
    if length(eqs) == length(states(b))
        xs = states(b)
        d = Differential(get_iv(b))
        eqs = [d(xs[i]) ~ eq for (i,eq) in enumerate(eqs)]
    end

    # Build a basis
    res_ = Basis(
        eqs, states(b),
        parameters = [parameters(b); p_], iv = get_iv(b),
        controls = controls(b), observed = observed(b),
        name = gensym(:Basis),
        eval_expression = eval_expression
    )

    sparsity = norm(Ξ, 0)
    sparsities = map(i->norm(i, 0), eachcol(Ξ))

    retcode = size(Ξ, 2) == size(prob.DX, 1) ? :success : :incomplete
    pnew = !isempty(parameters(b)) ? [prob.p; ps] : ps
    X = get_target(prob)
    Y = res_(get_oop_args(prob)...)

    # Build the metrics
    sparsity = norm(Ξ, 0)
    sparsities = map(i->norm(i, 0), eachcol(Ξ))
    inds = sparsities .> zero(eltype(X))
    error = norm(X[inds, :]-Y, 2)
    k = free_parameters(res_)
    aic = AICC(k, X[inds, :], Y)
    errors = zeros(eltype(X), sum(inds))
    aiccs = zeros(eltype(X), sum(inds))
    j = 1
    for i in 1:size(X,1)
        if inds[i]
            errors[i] = norm(X[i,:].-Y[j,:],2)
            aiccs[i] = AICC(k, X[i:i, :], Y[j:j,:])
            j += 1
        end
    end

    metrics = (
        Sparsity = sparsity,
        Error = error,
        AICC = aic,
        Sparsities = sparsities,
        Errors = errors,
        AICCs = aiccs,
    )

    inputs = (
        Problem = prob,
        Basis = b,
    )

    return DataDrivenSolution(
        res_, retcode, pnew, opt, Ξ, inputs, metrics
    )
end

function build_solution(prob::DataDrivenProblem, Ξ::AbstractMatrix, opt::Optimize.AbstractSubspaceOptimizer,
    b::Basis, implicits::Vector{Num}; eval_expression = false)

    if all(iszero(Ξ))
        @warn "Sparse regression failed! All coefficients are zero."
        return DataDrivenSolution(
        nothing , :failed, nothing, opt, Ξ, (Problem = prob, Basis = b, nothing),
    )
    end

    eqs, ps, p_ = build_parametrized_eqs(Ξ, b)
    eqs = [0 .~ eq for eq in eqs]
    
    # Build a basis
    res_ = Basis(
        collect(eqs), states(b),
        parameters = [parameters(b); p_], iv = get_iv(b),
        controls = controls(b), observed = observed(b),
        name = gensym(:Basis),
        eval_expression = eval_expression
    )

    sparsity = norm(Ξ, 0)
    sparsities = map(i->norm(i, 0), eachcol(Ξ))

    retcode = size(Ξ, 2) == size(prob.DX, 1) ? :success : :incomplete
    pnew = !isempty(parameters(b)) ? [prob.p; ps] : ps
    Y = res_([prob.X; prob.DX], pnew, prob.t, prob.U)

    # Build the metrics
    sparsity = norm(Ξ, 0)
    sparsities = map(i->norm(i, 0), eachcol(Ξ))
    inds = sparsities .> zero(eltype(Y))
    error = norm(Y, 2)
    k = free_parameters(res_)
    aic = AICC(k, Y, zero(Y))
    errors = zeros(eltype(Y), sum(inds))
    aiccs = zeros(eltype(Y), sum(inds))
    j = 1
    for i in 1:size(Y,1)
        if inds[i]
            errors[i] = norm(Y[j,:],2)
            aiccs[i] = AICC(k, Y[j:j,:], zero(Y[j:j,:]))
            j += 1
        end
    end

    metrics = (
        Sparsity = sparsity,
        Error = error,
        AICC = aic,
        Sparsities = sparsities,
        Errors = errors,
        AICCs = aiccs,
    )

    inputs = (
        Problem = prob,
        Basis = b,
        implicit_variables = implicits,
    )

    return DataDrivenSolution(
        res_, retcode, pnew, opt, Ξ, inputs, metrics
    )
end

function _round!(x::AbstractArray{T, N}, digits::Int) where {T, N}
    for i in eachindex(x)
        x[i] = round(x[i], digits = digits)
    end
    return x
end

function build_solution(prob::DataDrivenProblem, k, C, B, Q, P, inds, b::AbstractBasis, alg::AbstractKoopmanAlgorithm; digits::Int = 10, eval_expression = false)
    
    # Build parameterized equations, inds indicate the location of basis elements containing an input
    Ξ = zeros(eltype(B), size(C,2), length(b))

    Ξ[:, inds] .= real.(Matrix(k))
    if !isempty(B)
        Ξ[:, .! inds] .= B
    end

    # Transpose because of the nature of build_parametrized_eqs
    if !eval_expression
        eqs, ps, p_ = build_parametrized_eqs(_round!(C*Ξ, digits)', b)
    else
        K̃ = _round!(C*Ξ, digits)
        eqs = K̃*Num[states(b); controls(b)]
        p_ = []
        ps = [K̃...]
    end

    # Build the lhs
    if length(eqs) == length(states(b))
        xs = states(b)
        d = Differential(get_iv(b))
        eqs = [d(xs[i]) ~ eq for (i,eq) in enumerate(eqs)]
    end

    res_ = Koopman(eqs, states(b),
        parameters = [parameters(b); p_],
        controls = controls(b), iv = get_iv(b),
        K = k, C = C, Q = Q, P = P, lift = b.f,
        is_discrete = is_discrete(prob),
        eval_expression = eval_expression)

    retcode = :success
    X = get_target(prob)
    X_, p_, t, U = get_oop_args(prob)

    pnew = !isempty(parameters(b)) ? [p_; ps] : ps

    if !eval_expression
        # Equation space
        Y = res_(X_, pnew, t, U)
    else
        Y = K̃*b(X_, p_, t, U)
    end

    # Build the metrics
    error = norm(X-Y, 2)
    k = free_parameters(res_)
    aic = AICC(k, X, Y)
    errors = zeros(eltype(X), sum(inds))
    aiccs = zeros(eltype(X), sum(inds))
    j = 1
    for i in 1:size(X,1)
        errors[i] = norm(X[i,:].-Y[i,:],2)
        aiccs[i] = AICC(k, X[i:i, :], Y[i:i,:])
    end

    metrics = (
        Error = error,
        AICC = aic,
        Errors = errors,
        AICCs = aiccs,
    )

    inputs = (
        Problem = prob,
        Basis = b,
    )

    return DataDrivenSolution(
        res_, retcode, pnew, alg, Ξ, inputs, metrics
    )
end
