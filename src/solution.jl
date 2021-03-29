
struct DataDrivenSolution{R <: AbstractBasis, S, P , A, O,M} <: AbstractDataDrivenSolution
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

Generate an mapping of the parameter values and symbolic representation useable
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
    println(io, "Solution with $(length(r.res.eqs)) equations and $(length(r.ps)) parameters.")
    println(io, "Returncode: $(r.retcode)")
    println(io, "Sparsity: $(r.metrics.Sparsity)")
    println(io, "L2 Norm Error: $(r.metrics.Error)")
    println(io, "AICC: $(r.metrics.AICC)")
end

Base.print(io::IO, r::DataDrivenSolution) = summary(io, r)
Base.show(io::IO, r::DataDrivenSolution) = is_implicit(r) ? show(io,"Implicit Result") : show(io,"Explicit Result")


function build_parametrized_eqs(X::AbstractMatrix, b::Basis)
    # Create additional variables
    sp = Int(norm(X, 0))
    sps = norm.(eachcol(X), 0)
    inds = sps .> zero(eltype(X))
    pl = length(parameters(b))
    @parameters p[(pl+1):(pl+sp)]
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
function build_solution(prob::DataDrivenProblem, Ξ::AbstractMatrix, opt::Optimize.AbstractOptimizer, b::Basis)
    eqs, ps, p_ = build_parametrized_eqs(Ξ, b)

    # Build the lhs
    if length(eqs) == length(states(b))
        xs = states(b)
        d = Differential(independent_variable(b))
        eqs = [d(xs[i]) ~ eq for (i,eq) in enumerate(eqs)]
    end

    # Build a basis
    res_ = Basis(
        eqs, states(b),
        parameters = [parameters(b); p_], iv = independent_variable(b),
        controls = controls(b), observed = observed(b),
        name = gensym(:Basis)
    )

    sparsity = norm(Ξ, 0)
    sparsities = map(i->norm(i, 0), eachcol(Ξ))

    retcode = size(Ξ, 2) == size(prob.DX, 1) ? :sucess : :incomplete
    pnew = [prob.p; ps]
    X = prob.DX
    Y = res_(prob.X, pnew, prob.t, prob.U)

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
    b::Basis, implicits::Vector{Num})
    eqs, ps, p_ = build_parametrized_eqs(Ξ, b)
    eqs = [0 .~ eq for eq in eqs]

    # Build a basis
    res_ = Basis(
        eqs, states(b),
        parameters = [parameters(b); p_], iv = independent_variable(b),
        controls = controls(b), observed = observed(b),
        name = gensym(:Basis)
    )

    sparsity = norm(Ξ, 0)
    sparsities = map(i->norm(i, 0), eachcol(Ξ))

    retcode = size(Ξ, 2) == size(prob.DX, 1) ? :sucess : :incomplete
    pnew = [prob.p; ps]
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
