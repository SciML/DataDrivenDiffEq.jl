function __repeated_solve(X, Y, options)
    B = Y * X'
    prob = LinearProblem(X * X', B[1, :])
    linear_prob = LinearSolve.init(prob)
    @unpack linsolve, abstol, reltol, maxiters, Pl, Pr, verbose = options
    Pl = isnothing(Pl) ? LinearSolve.Identity() : Pl
    Pr = isnothing(Pr) ? LinearSolve.Identity() : Pr
    K = zeros(eltype(X), size(B, 1), size(X, 1))
    res = solve(linear_prob, linsolve, abstol = abstol, reltol = reltol,
                maxiters = maxiters, Pl = Pl, Pr = Pr, verbose = verbose)
    K[1, :] .= res.u
    for i in 2:size(B, 1)
        linear_prob = LinearSolve.set_b(res.cache, B[i, :])
        res = solve(linear_prob, linsolve, abstol = abstol, reltol = reltol,
                    maxiters = maxiters, Pl = Pl, Pr = Pr, verbose = verbose)
        K[i, :] .= res.u
    end
    return K
end

function (alg::AbstractKoopmanAlgorithm)(X, Y, B, Z, inds;
                                         options::DataDrivenCommonOptions = DataDrivenCommonOptions())
    if all(inds)
        K, B = __apply_alg(X, Y, options)
    elseif isempty(b)
        K, B = __apply_alg(X[inds, :], Y[inds, :], X[.!inds, :], options)
    else
        K, B = __apply_alg(X[inds, :], Y[inds, :], X[.!inds, :], b, options)
    end

    Q = Y[inds, :] * X'
    P = X * X'
    C = __repeated_solve(Z, Y[inds, :], options)

    return KoopmanResult(K, B, C, P, Q, :Default)
end

function __apply_alg(x::AbstractKoopmanAlgorithm, X::AbstractArray, Y::AbstractArray,
                     U::AbstractArray,
                     B::AbstractArray)
    K, _ = __apply_alg(x, X, Y - B * U)
    return K, B
end

"""
$(TYPEDEF)

Approximates the [`Koopman`](@ref) by solving the linear system

```julia
Y = K X
```

where `Y` and `X` are data matrices. Returns an [`KoopmanResult`](@ref).
"""
struct DMDPINV <: AbstractKoopmanAlgorithm end

function __apply_alg(::DMDPINV, X::AbstractArray, Y::AbstractArray,
                     options::DataDrivenCommonOptions)
    K = __repeated_solve(X, Y, options)
    return K, []
end

function __apply_alg(alg::DMDPINV, X::AbstractArray, Y::AbstractArray, U::AbstractArray,
                     options::DataDrivenCommonOptions)
    nx, m = size(X)
    X̃ = [X; U]
    K̃, _ = __apply_alg(alg, X̃, Y, options)

    return K̃[:, 1:nx], K̃[:, (nx + 1):end]
end
