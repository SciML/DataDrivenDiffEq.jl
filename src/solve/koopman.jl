
function DiffEqBase.solve(prob::DataDrivenProblem{dType}, alg::AbstractKoopmanAlgorithm;
    kwargs...) where {dType <: Number}
    # Check the validity
    @assert is_valid(prob) "The problem seems to be ill-defined. Please check the problem definition."


    if has_inputs(prob)
        if is_continuous(prob)
            X = vcat(prob.X, prob.U)
            Y = vcat(prob.DX, prob.U)
        else
            X = vcat(prob.X, prob.U[:, 1:end-1])
            Y = vcat(prob.DX, prob.U[:, 2:end])
        end
        n_u = size(prob.U, 1)
        n_x = size(prob.X, 1)
    else
        X = prob.X
        Y = prob.DX
        n_u = 0
        n_x = size(prob.X, 1)
    end

    # The input maps
    k = alg(X, Y)

    if n_u > 0
        # Split the eigendecomposition
        x_v, x_vecs = k.values[1:n_x], k.vectors[1:n_x, 1:n_x]
        u_v, u_vecs = k.values[(n_x+1):(n_x+n_u)], k.vectors[n_x+1:end,(n_x+1):(n_x+n_u)]
        return k
        k = Eigen(x_v, x_vecs)
        B = Matrix(Eigen(u_v, u_vecs))
    else
        B = zeros(eltype(X), 0, 0)
    end

    # Updateable for all measurements
    Q = Y*X'
    P = Y*X'

    return k, B

    return Koopman(k; B = B, Q = Q, P = P)
end

function DiffEqBase.solve(prob::DataDrivenProblem{dType}, b::Basis, alg::AbstractKoopmanAlgorithm;
    kwargs...) where {dType <: Number}
    # Check the validity
    @assert is_valid(prob) "The problem seems to be ill-defined. Please check the problem definition."

    Ψ₀ = b(prob.X, prob.p, prob.t[1:end-1], prob.U[:, 1:end-1])
    Ψ₁ = b(prob.DX, prob.p, prob.t[2:end], prob.U[:, 2:end])
    k = alg(Ψ₀, Ψ₁)

    # Outpumap
    C = prob.X*pinv(Ψ₀)

    Q = Ψ₁*Ψ₀'
    P = Ψ₀*Ψ₀'
    return Koopman(Num[eq.rhs for eq in equations(b)], Num.(states(b)), K = k, C = C, Q = Q, P = P)
end
