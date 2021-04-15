
function DiffEqBase.solve(prob::DataDrivenProblem{dType}, alg::AbstractKoopmanAlgorithm;
    B::AbstractArray = [], digits::Int = 10,
    kwargs...) where {dType <: Number}
    # Check the validity
    @assert is_valid(prob) "The problem seems to be ill-defined. Please check the problem definition."

    X = prob.X
    DX = prob.DX

    # The input maps
    if !has_inputs(prob)
        k = alg(X, DX)
        B = Matrix{dType}(undef, 0, 0)
        # Updateable for all measurements
        Q = DX*X'
        P = DX*X'

    else
        if isempty(B)
            k, B = alg(X, DX, prob.U)
        else
            k, B = alg(X, DX, prob.U, B)
        end
        # Updateable for all measurements
        Q = DX*[X;prob.U]'
        P = DX*[X;prob.U]'
    end


    return Koopman(k; B = B, Q = Q, P = P, digits = digits)
end

function DiffEqBase.solve(prob::DataDrivenProblem{dType}, b::Basis, alg::AbstractKoopmanAlgorithm; digits::Int = 10,
    kwargs...) where {dType <: Number}
    # Check the validity
    @assert is_valid(prob) "The problem seems to be ill-defined. Please check the problem definition."

    X, p, t, U = get_oop_args(prob)
    DX = prob.DX

    Ψ₀ = b(X, p, t, U)
    Ψ₁ = similar(Ψ₀)

    if is_continuous(prob)
        # Generate the differential mapping
        J = jacobian(b)
        for i in 1:size(DX, 2)
            Ψ₁[:, i] .= J(X[:, i], p, t[i])*DX[:, i]
        end
    else
        b(Ψ₁, DX, p, t, U)
    end

    k = alg(Ψ₀, Ψ₁)

    # Outpumap
    C = prob.X / Ψ₀

    Q = Ψ₁*Ψ₀'
    P = Ψ₀*Ψ₀'

    return Koopman(Num[eq.rhs for eq in equations(b)], Num.(states(b)), K = k, C = C, Q = Q, P = P, digits = digits)
end
