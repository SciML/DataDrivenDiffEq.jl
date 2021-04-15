
function DiffEqBase.solve(prob::DataDrivenProblem{dType}, alg::AbstractKoopmanAlgorithm;
    B::AbstractArray = [], digits::Int = 10, operator_only::Bool = false,
    kwargs...) where {dType <: Number}
    # Check the validity
    @assert is_valid(prob) "The problem seems to be ill-defined. Please check the problem definition."

    X = prob.X
    DX = prob.DX
    U = prob.U


    # Create a basis
    @variables x[1:size(X, 1)] u[1:size(U, 1)] t
    b = Basis([x; u], x, controls = u, iv = t)

    inds = BitVector([i<=size(X, 1) ? true : false for i in 1:size(X, 1)+size(U,1)])
    C = diagm(ones(dType, size(X,1)))

    # The input maps
    if !has_inputs(prob)
        k = alg(X, DX)
        B = Matrix{dType}(undef, 0, 0)
        # Updateable for all measurements
        Q = DX*X'
        P = DX*X'

    else
        if isempty(B)
            k, B = alg(X, DX, U)
        else
            k, B = alg(X, DX, U, B)
        end
        # Updateable for all measurements
        Q = DX*[X;U]'
        P = DX*[X;U]'

    end

    operator_only && return (K = k, C = C, B = B, Q = Q, P = P)

    return build_solution(prob, k, C, B, Q, P, inds, b, alg, digits = digits)
end

function DiffEqBase.solve(prob::DataDrivenProblem{dType}, b::Basis, alg::AbstractKoopmanAlgorithm;
    digits::Int = 10, operator_only::Bool = false,
    kwargs...) where {dType <: Number}
    # Check the validity
    @assert is_valid(prob) "The problem seems to be ill-defined. Please check the problem definition."

    X, p, t, U = get_oop_args(prob)
    DX = prob.DX

    Ψ₀ = b(X, p, t, U)
    Ψ₁ = similar(Ψ₀)

    # Find the indexes of the control states
    if has_inputs(prob) && !isempty(controls((b)))
        inds = .! _isin(Num.(controls(b)), [eq.rhs for eq in equations(b)])[1,:]
    else
        inds = [true for i in 1:length(b)]
    end

    if is_continuous(prob)
        # Generate the differential mapping
        J = jacobian(b)
        if has_inputs(prob)
            for i in 1:size(DX, 2)
                Ψ₁[:, i] .= J(X[:, i], p, t[i],U[:, i])*DX[:, i]
            end
        else
            for i in 1:size(DX, 2)
                Ψ₁[:, i] .= J(X[:, i], p, t[i])*DX[:, i]
            end
        end
    else
        has_inputs(prob) ? b(Ψ₁, DX, p, t, U) : b(Ψ₁, DX, p, t)
    end

    if has_inputs(prob)

        k, B = alg(Ψ₀[inds, :], Ψ₁[inds, :], Ψ₀[.!inds, :])


        Q = Ψ₁[inds, :]*Ψ₀'
        P = Ψ₀*Ψ₀'

    else
        k = alg(Ψ₀, Ψ₁)

        Q = Ψ₁*Ψ₀'
        P = Ψ₀*Ψ₀'
        B = zeros(dType, 0, 0)

    end

    # Outpumap -> just the state dependent
    C = prob.DX / Ψ₁[inds,:]

    operator_only && return (K = k, C = C, B = B, Q = Q, P = P)

    return build_solution(prob, k, C, B, Q, P, inds, b, alg, digits = digits)
end
