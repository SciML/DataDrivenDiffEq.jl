## Discrete Time
function DiffEqBase.solve(prob::AbstractDiscreteProb{dType, true}, alg::AbstractKoopmanAlgorithm;
    B::AbstractArray = [], digits::Int = 10, operator_only::Bool = false,
    eval_expression = false,
    kwargs...) where {dType <: Number}
    # Check the validity
    @assert is_valid(prob) "The problem seems to be ill-defined. Please check the problem definition."

    X = prob.X[:,1:end-1]
    DX = prob.X[:,2:end]


    # Create a basis
    @variables x[1:size(X, 1)] t
    b = Basis(x, x,  iv = t)

    C = diagm(ones(dType, size(X,1)))

    k = alg(X, DX)
    B = Matrix{dType}(undef, 0, 0)
    # Updateable for all measurements
    Q = DX*X'
    P = DX*X'

    operator_only && return (K = k, C = C, B = B, Q = Q, P = P)

    return DataDrivenSolution(prob, k, C, B, Q, P, BitVector((true for i in 1:size(X,1))), b, alg, digits = digits, eval_expression = eval_expression)
end


function DiffEqBase.solve(prob::AbstractDiscreteProb{dType, false}, alg::AbstractKoopmanAlgorithm;
    B::AbstractArray = [], digits::Int = 10, operator_only::Bool = false,
    eval_expression = false,
    kwargs...) where {dType <: Number}
    # Check the validity
    @assert is_valid(prob) "The problem seems to be ill-defined. Please check the problem definition."

    X = prob.X[:,1:end-1]
    DX = prob.X[:,2:end]
    U = prob.U[:, 1:end-1]


    # Create a basis
    @variables x[1:size(X, 1)] u[1:size(U, 1)] t
    b = Basis([x; u], x, controls = u, iv = t)

    inds = BitVector([ones(Bool, size(X,1)); zeros(Bool, size(U,1))])
    C = diagm(ones(dType, size(X,1)))

    # The input maps

    if isempty(B)
        k, B = alg(X, DX, U)
    else
        k, B = alg(X, DX, U, B)
    end
    # Updateable for all measurements
    Q = DX*[X;U]'
    P = DX*[X;U]'

    operator_only && return (K = k, C = C, B = B, Q = Q, P = P)

    return DataDrivenSolution(prob, k, C, B, Q, P, inds, b, alg, digits = digits, eval_expression = eval_expression)
end

function DiffEqBase.solve(prob::AbstractDiscreteProb{dType, true}, b::Basis, alg::AbstractKoopmanAlgorithm;
    digits::Int = 10, operator_only::Bool = false,
    eval_expression = false,
    kwargs...) where {dType <: Number}
    # Check the validity
    @assert is_valid(prob) "The problem seems to be ill-defined. Please check the problem definition."

    X = prob.X[:,1:end-1]
    DX = prob.X[:,2:end]
    p = prob.p
    t = prob.t

    Ψ₀ = b(X, p, t[1:end-1])
    Ψ₁ = b(DX, p, t[2:end])

    k = alg(Ψ₀, Ψ₁)

    Q = Ψ₁*Ψ₀'
    P = Ψ₀*Ψ₀'
    B = zeros(dType, 0, 0)

    # Outpumap -> just the state dependent
    C = DX / Ψ₁

    operator_only && return (K = k, C = C, B = B, Q = Q, P = P)

    return DataDrivenSolution(prob, k, C, B, Q, P, BitVector((true for i in 1:size(Ψ₀,1))), b, alg, digits = digits, eval_expression = eval_expression)
end


function DiffEqBase.solve(prob::AbstractDiscreteProb{dType, false}, b::Basis, alg::AbstractKoopmanAlgorithm;
    digits::Int = 10, operator_only::Bool = false,
    eval_expression = false,
    kwargs...) where {dType <: Number}
    # Check the validity
    @assert is_valid(prob) "The problem seems to be ill-defined. Please check the problem definition."

    X = prob.X[:,1:end-1]
    DX = prob.X[:,2:end]
    U = prob.U[:, 1:end-1]
    p = prob.p
    t = prob.t

    Ψ₀ = b(X, p, t[1:end-1], U)
    Ψ₁ = b(DX, p,t[2:end], U)

    # Find the indexes of the control states
    inds = .! is_dependent(map(eq->Num(eq.rhs),equations(b)), Num.(controls(b)))[1,:]

    k, B = alg(Ψ₀[inds, :], Ψ₁[inds, :], Ψ₀[.!inds, :])

    Q = Ψ₁[inds, :]*Ψ₀'
    P = Ψ₀*Ψ₀'

    # Outpumap -> just the state dependent
    C = DX / Ψ₁[inds,:]

    operator_only && return (K = k, C = C, B = B, Q = Q, P = P)

    return DataDrivenSolution(prob, k, C, B, Q, P, inds, b, alg, digits = digits, eval_expression = eval_expression)
end

## Continouos Time
function DiffEqBase.solve(prob::AbstracContProb{dType, true}, alg::AbstractKoopmanAlgorithm;
    B::AbstractArray = [], digits::Int = 10, operator_only::Bool = false,
    eval_expression = false,
    kwargs...) where {dType <: Number}
    # Check the validity
    @assert is_valid(prob) "The problem seems to be ill-defined. Please check the problem definition."

    X = prob.X
    DX = prob.DX


    # Create a basis
    @variables x[1:size(X, 1)] t
    b = Basis(x, x,  iv = t)

    C = diagm(ones(dType, size(X,1)))

    k = alg(X, DX)
    B = Matrix{dType}(undef, 0, 0)
    # Updateable for all measurements
    Q = DX*X'
    P = DX*X'

    operator_only && return (K = k, C = C, B = B, Q = Q, P = P)

    return DataDrivenSolution(prob, k, C, B, Q, P, BitVector((true for i in 1:size(X,1))), b, alg, digits = digits, eval_expression = eval_expression)
end


function DiffEqBase.solve(prob::AbstracContProb{dType, false}, alg::AbstractKoopmanAlgorithm;
    B::AbstractArray = [], digits::Int = 10, operator_only::Bool = false,
    eval_expression = false,
    kwargs...) where {dType <: Number}
    # Check the validity
    @assert is_valid(prob) "The problem seems to be ill-defined. Please check the problem definition."

    X = prob.X
    DX = prob.DX
    U = prob.U


    # Create a basis
    @variables x[1:size(X, 1)] u[1:size(U, 1)] t
    b = Basis([x; u], x, controls = u, iv = t)

    inds = BitVector([ones(Bool, size(X,1)); zeros(Bool, size(U,1))])
    C = diagm(ones(dType, size(X,1)))

    # The input maps

    if isempty(B)
        k, B = alg(X, DX, U)
    else
        k, B = alg(X, DX, U, B)
    end
    # Updateable for all measurements
    Q = DX*[X;U]'
    P = DX*[X;U]'

    operator_only && return (K = k, C = C, B = B, Q = Q, P = P)

    return DataDrivenSolution(prob, k, C, B, Q, P, inds, b, alg, digits = digits, eval_expression = eval_expression)
end

function DiffEqBase.solve(prob::AbstracContProb{dType, true}, b::Basis, alg::AbstractKoopmanAlgorithm;
    digits::Int = 10, operator_only::Bool = false,
    eval_expression = false,
    kwargs...) where {dType <: Number}
    # Check the validity
    @assert is_valid(prob) "The problem seems to be ill-defined. Please check the problem definition."

    X = prob.X
    DX = prob.DX
    p = prob.p
    t = prob.t

    Ψ₀ = b(X, p, t)
    Ψ₁ = similar(Ψ₀)

    J = jacobian(b)

    for i in 1:size(DX, 2)
        Ψ₁[:, i] .= J(X[:, i], p, t[i])*DX[:, i]
    end

    k = alg(Ψ₀, Ψ₁)

    Q = Ψ₁*Ψ₀'
    P = Ψ₀*Ψ₀'
    B = zeros(dType, 0, 0)

    # Outpumap -> just the state dependent
    C = prob.DX / Ψ₁

    operator_only && return (K = k, C = C, B = B, Q = Q, P = P)

    return DataDrivenSolution(prob, k, C, B, Q, P, BitVector((true for i in 1:size(Ψ₀,1))), b, alg, digits = digits, eval_expression = eval_expression)
end


function DiffEqBase.solve(prob::AbstracContProb{dType, false}, b::Basis, alg::AbstractKoopmanAlgorithm;
    digits::Int = 10, operator_only::Bool = false,
    eval_expression = false,
    kwargs...) where {dType <: Number}
    # Check the validity
    @assert is_valid(prob) "The problem seems to be ill-defined. Please check the problem definition."

    X = prob.X
    DX = prob.DX
    U = prob.U
    p = prob.p
    t = prob.t

    Ψ₀ = b(X, p, t, U)
    Ψ₁ = similar(Ψ₀)

    J = jacobian(b)

    for i in 1:size(DX, 2)
        Ψ₁[:, i] .= J(X[:, i], p, t[i], U[:,i])*DX[:, i]
    end

    # Find the indexes of the control states
    inds = .! is_dependent(map(eq->Num(eq.rhs),equations(b)), Num.(controls(b)))[1,:]

    k, B = alg(Ψ₀[inds, :], Ψ₁[inds, :], Ψ₀[.!inds, :])

    Q = Ψ₁[inds, :]*Ψ₀'
    P = Ψ₀*Ψ₀'

    # Outpumap -> just the state dependent
    C = prob.DX / Ψ₁[inds,:]

    operator_only && return (K = k, C = C, B = B, Q = Q, P = P)

    return DataDrivenSolution(prob, k, C, B, Q, P, inds, b, alg, digits = digits, eval_expression = eval_expression)
end