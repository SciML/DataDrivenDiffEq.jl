import DataDrivenDiffEq.is_dependent

struct KoopmanProblem{X, Y, U, C, PR, B, TR, TS, P, O}
    x::X
    y::Y
    b::U
    inds::C
    prob::PR
    basis::B
    train::TR
    test::TS
    alg::P
    options::O
    eval_expression::Bool
end

# DMD-Like
function CommonSolve.init(prob::AbstractDataDrivenProblem,
                          alg::AbstractKoopmanAlgorithm, args...; kwargs...) where {N, C, P}
    # Build a basis
    s_x = size(prob.X, 1)
    s_u = size(prob.U, 1)

    x = [Symbolics.variable(:x, i) for i in 1:s_x]
    u = [Symbolics.variable(:u, i) for i in 1:s_u]
    t = Symbolics.variable(:t)

    b = Basis([x; u], x, controls = u, iv = t)

    init(prob, b, alg, args...; kwargs...)
end

function CommonSolve.init(prob::DataDrivenDiffEq.ABSTRACT_DIRECT_PROB, b::AbstractBasis,
                          alg::AbstractKoopmanAlgorithm, args...; kwargs...)
    throw(ArgumentError("DirectDataDrivenProblems can not be solved via Koopman based inference. Please use a different algorithm."))
end

# All (g(E))DMD like
function CommonSolve.init(prob::DataDrivenDiffEq.ABSTRACT_DISCRETE_PROB{N, C},
                          b::AbstractBasis, alg::A,
                          args...; B = [], eval_expression = false,
                          kwargs...) where {N, C, A <: AbstractKoopmanAlgorithm}
    @is_applicable prob

    @unpack X, p, t, U = prob

    x = b(X[:, 1:(end - 1)], p, t[1:(end - 1)], U[:, 1:(end - 1)])
    y = b(X[:, 2:end], p, t[2:end], U[:, 2:end])

    if !isempty(controls(b))
        inds = .!is_dependent(map(eq -> Num(eq.rhs), equations(b)), Num.(controls(b)))[1, :]
    else
        inds = ones(Bool, length(b))
    end

    options = DataDrivenCommonOptions(alg, N; kwargs...)

    @unpack sampler = options

    train, test = sampler(prob)

    return KoopmanProblem(x, y, B, inds, prob, b, train, test, alg, options,
                          eval_expression)
end

function CommonSolve.init(prob::DataDrivenDiffEq.ABSTRACT_CONT_PROB{N, C}, b::AbstractBasis,
                          alg::A, args...;
                          B = [], eval_expression = false,
                          kwargs...) where {N, C, A <: AbstractKoopmanAlgorithm}
    @is_applicable prob

    @unpack DX, X, p, t, U = prob

    x = b(prob)

    y = similar(x)

    J = jacobian(b)

    if !isempty(U)
        # Find the indexes of the control states
        inds = .!is_dependent(map(eq -> Num(eq.rhs), equations(b)), Num.(controls(b)))[1, :]
        for i in 1:length(prob)
            y[:, i] .= J(X[:, i], p, t[i], U[:, i]) * DX[:, i]
        end
    else
        inds = ones(Bool, length(b))
        for i in 1:length(prob)
            y[:, i] .= J(X[:, i], p, t[i]) * DX[:, i]
        end
    end

    options = DataDrivenCommonOptions(alg, N; kwargs...)

    @unpack sampler = options

    train, test = sampler(prob)

    return KoopmanProblem(x, y, B, inds, prob, b, train, test, alg, options,
                          eval_expression)
end

function CommonSolve.solve!(k::KoopmanProblem)
    @unpack x, y, b, inds, prob, basis, train, test, alg, options, eval_expression = k
    @unpack normalize, denoise, sampler, maxiter, abstol, reltol, verbose, progress, f, g, digits, kwargs = options

    z = get_target(prob)

    results = []

    for (i, t) in enumerate(train)
        res = derive_operator(alg, x[:, t], y[:, t], b, z[:, t], inds)
        push!(results, res)
    end

    return results
end
