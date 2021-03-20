
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

(r::DataDrivenSolution)(args...) = r.res(args...)

function build_parametrized_eqs(X::AbstractMatrix, b::Basis)
    # Create additional variables
    sp = Int(norm(X, 0))
    sps = norm.(eachcol(X), 0)
    inds = sps .> zero(eltype(X))
    pl = length(parameters(b))
    @variables p[(pl+1):(pl+sp)]
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
