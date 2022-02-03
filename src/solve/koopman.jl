struct KoopmanProblem{X,Y,U,C,PR,B,TR,TS,P,O}
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
struct KoopmanResult{O,S, E, F, A, P}
    k::O
    sets::S
    error::E
    folds::F
    alg::A
    options::P
end

## Apply the problem to get the operator

# DMD-Like
function CommonSolve.init(prob::AbstractDataDrivenProblem{N,C,P}, alg::AbstractKoopmanAlgorithm, args...; kwargs...) where {N,C,P}
    @info "Koopman"
    # Build a basis
    s_x = size(prob.X,1)
    s_u = size(prob.U,1)

    x = [Symbolics.variable(:x, i) for i in 1:s_x]
    u = [Symbolics.variable(:u, i) for i in 1:s_u]
    t = Symbolics.variable(:t)
    
    b = Basis([x; u], x, controls = u, iv = t)
    
    init(prob, b, alg, args...; B = B, kwargs...)
end

# All (g(E))DMD like
function CommonSolve.init(prob::AbstractDiscreteProb{N,C}, b::AbstractBasis, alg::A, args...; B = [], eval_expression = false,  kwargs...) where {N,C, A <: AbstractKoopmanAlgorithm}
    @info "Koopman"
    @is_applicable prob 

    @unpack X,p,t,U = prob

    x = b(X[:,1:end-1], p, t[1:end-1], U[:,1:end-1])
    y = b(X[2:end-1], p, t[2:end], U[:, 2:end])

    inds = .! is_dependent(map(eq->Num(eq.rhs),equations(b)), Num.(controls(b)))[1,:]

    options = DataDrivenCommonOptions(alg, N; kwargs...)
    
    @unpack sampler = options

    train, test = sampler(prob)

    return KoopmanProblem(
        x, y, B, inds, prob, b, train, test, alg, options, eval_expression
    )
end

function CommonSolve.init(prob::AbstracContProb{N,C}, b::AbstractBasis, alg::A, args...; B = [], eval_expression = false,  kwargs...) where {N,C, A <: AbstractKoopmanAlgorithm}
    @is_applicable prob 

    @unpack DX,X,p,t,U = prob

    x = b(prob)

    y = similar(x)

    J = jacobian(b)

    for i in 1:length(prob)
       y[:, i] .= J(X[:, i], p, t[i], U[:, i])*DX[:, i]
    end

    options = DataDrivenCommonOptions(alg, N; kwargs...)
    
    @unpack sampler = options

    # Right now just ignore this 
    #train , test = nothing, nothing
    train, test = sampler(prob)

    # Find the indexes of the control states
    inds = .! is_dependent(map(eq->Num(eq.rhs),equations(b)), Num.(controls(b)))[1,:]

    return KoopmanProblem(
        x, y, B, inds, prob, b, train, test, alg, options, eval_expression
    )
end

function derive_operator(alg, x, y, b, z, inds)
    if all(inds)
        K, B = alg(x,y)
        Q = y[inds, :]*x'
        P = x*x'
        C = z / y[inds, :]
    elseif isempty(b) 
        K, B =  alg(x[inds, :], y[inds, :], x[.! inds, :])
        Q = y[inds, :]*x'
        P = x*x'
        C = z / y[inds, :]
    else
        K, B =  alg(x[inds, :], y[inds, :], x[.! inds, :], b)
        Q = y[inds, :]*x'
        P = x*x'
        C = z / y[inds, :]
    end
    return K, B, C, P, Q
end

function operator_error(f, g)
    (x,y,K,B,C,P,Q,inds) -> begin
        k_ = Matrix(K)
        isempty(B) && g(f(k_*x, C, y))
        g(f(k_*x[inds, :]+B*x[.! inds, :], C, y))
    end
end

function CommonSolve.solve!(k::KoopmanProblem)
    @info "Koopman"
    @unpack x, y, b, inds, prob, basis, train, test, alg, options, eval_expression = k
    @unpack normalize, denoise, sampler, maxiter, abstol, reltol, verbose, progress,f,g, kwargs = options
    
    z = get_target(prob)
    xₜ = x[:, test]
    zₜ = z[:, test]

    trainerror = zeros(eltype(z), length(train), length(train))
    testerror = zeros(eltype(z), length(train))
    fg = operator_error(f, g)
    ops = []

    for (i,t) in enumerate(train)
        op = derive_operator(alg, x[:, t], y[:, t], b, z[:, t], inds)

        push!(ops, op)

        testerror[i] = fg(xₜ, zₜ, op..., inds)

        for (j, tt) in enumerate(train)
            trainerror[i, j] = fg(x[:, tt], z[:, tt], op..., inds)
        end
    end

    sol = KoopmanResult(ops,(train,test),testerror, trainerror, alg, options)
    return sol
    return K, B, C
end


## Discrete Time
#function DiffEqBase.solve(prob::AbstractDiscreteProb{dType, true}, alg::AbstractKoopmanAlgorithm;
#    B::AbstractArray = [], digits::Int = 10, operator_only::Bool = false,
#    eval_expression = false,
#    kwargs...) where {dType <: Number}
#    # Check the validity
#    @assert is_valid(prob) "The problem seems to be ill-defined. Please check the problem definition."
#
#    X = prob.X[:,1:end-1]
#    DX = prob.X[:,2:end]
#
#    # Create a basis
#    @variables x[1:size(X, 1)] t
#    b = Basis(x, x,  iv = t)
#
#    C = diagm(ones(dType, size(X,1)))
#
#    k = alg(X, DX)
#    B = Matrix{dType}(undef, 0, 0)
#    # Updateable for all measurements
#    Q = DX*X'
#    P = DX*X'
#
#    operator_only && return (K = k, C = C, B = B, Q = Q, P = P)
#
#    return DataDrivenSolution(prob, k, C, B, Q, P, BitVector((true for i in 1:size(X,1))), b, alg, digits = digits, eval_expression = eval_expression)
#end
#
#
#function DiffEqBase.solve(prob::AbstractDiscreteProb{dType, false}, alg::AbstractKoopmanAlgorithm;
#    B::AbstractArray = [], digits::Int = 10, operator_only::Bool = false,
#    eval_expression = false,
#    kwargs...) where {dType <: Number}
#    # Check the validity
#    @assert is_valid(prob) "The problem seems to be ill-defined. Please check the problem definition."
#
#    X = prob.X[:,1:end-1]
#    DX = prob.X[:,2:end]
#    U = prob.U[:, 1:end-1]
#
#
#    # Create a basis
#    @variables x[1:size(X, 1)] u[1:size(U, 1)] t
#    b = Basis([x; u], x, controls = u, iv = t)
#
#    inds = BitVector([ones(Bool, size(X,1)); zeros(Bool, size(U,1))])
#    C = diagm(ones(dType, size(X,1)))
#
#    # The input maps
#
#    if isempty(B)
#        k, B = alg(X, DX, U)
#    else
#        k, B = alg(X, DX, U, B)
#    end
#    # Updateable for all measurements
#    Q = DX*[X;U]'
#    P = DX*[X;U]'
#
#    operator_only && return (K = k, C = C, B = B, Q = Q, P = P)
#
#    return DataDrivenSolution(prob, k, C, B, Q, P, inds, b, alg, digits = digits, eval_expression = eval_expression)
#end
#
#function DiffEqBase.solve(prob::AbstractDiscreteProb{dType, true}, b::Basis, alg::AbstractKoopmanAlgorithm;
#    digits::Int = 10, operator_only::Bool = false,
#    eval_expression = false,
#    kwargs...) where {dType <: Number}
#    # Check the validity
#    @assert is_valid(prob) "The problem seems to be ill-defined. Please check the problem definition."
#
#    X = prob.X[:,1:end-1]
#    DX = prob.X[:,2:end]
#    p = prob.p
#    t = prob.t
#
#    Ψ₀ = b(X, p, t[1:end-1])
#    Ψ₁ = b(DX, p, t[2:end])
#
#    k = alg(Ψ₀, Ψ₁)
#
#    Q = Ψ₁*Ψ₀'
#    P = Ψ₀*Ψ₀'
#    B = zeros(dType, 0, 0)
#
#    # Outpumap -> just the state dependent
#    C = DX / Ψ₁
#
#    operator_only && return (K = k, C = C, B = B, Q = Q, P = P)
#
#    return DataDrivenSolution(prob, k, C, B, Q, P, BitVector((true for i in 1:size(Ψ₀,1))), b, alg, digits = digits, eval_expression = eval_expression)
#end
#
#
#function DiffEqBase.solve(prob::AbstractDiscreteProb{dType, false}, b::Basis, alg::AbstractKoopmanAlgorithm;
#    digits::Int = 10, operator_only::Bool = false,
#    eval_expression = false,
#    kwargs...) where {dType <: Number}
#    # Check the validity
#    @assert is_valid(prob) "The problem seems to be ill-defined. Please check the problem definition."
#
#    X = prob.X[:,1:end-1]
#    DX = prob.X[:,2:end]
#    U = prob.U[:, 1:end-1]
#    p = prob.p
#    t = prob.t
#
#    Ψ₀ = b(X, p, t[1:end-1], U)
#    Ψ₁ = b(DX, p,t[2:end], U)
#
#    # Find the indexes of the control states
#    inds = .! is_dependent(map(eq->Num(eq.rhs),equations(b)), Num.(controls(b)))[1,:]
#
#    k, B = alg(Ψ₀[inds, :], Ψ₁[inds, :], Ψ₀[.!inds, :])
#
#    Q = Ψ₁[inds, :]*Ψ₀'
#    P = Ψ₀*Ψ₀'
#
#    # Outpumap -> just the state dependent
#    C = DX / Ψ₁[inds,:]
#
#    operator_only && return (K = k, C = C, B = B, Q = Q, P = P)
#
#    return DataDrivenSolution(prob, k, C, B, Q, P, inds, b, alg, digits = digits, eval_expression = eval_expression)
#end
#
### Continouos Time
#function DiffEqBase.solve(prob::AbstracContProb{dType, true}, alg::AbstractKoopmanAlgorithm;
#    B::AbstractArray = [], digits::Int = 10, operator_only::Bool = false,
#    eval_expression = false,
#    kwargs...) where {dType <: Number}
#    # Check the validity
#    @assert is_valid(prob) "The problem seems to be ill-defined. Please check the problem definition."
#
#    X = prob.X
#    DX = prob.DX
#
#
#    # Create a basis
#    @variables x[1:size(X, 1)] t
#    b = Basis(x, x,  iv = t)
#
#    C = diagm(ones(dType, size(X,1)))
#
#    k = alg(X, DX)
#    B = Matrix{dType}(undef, 0, 0)
#    # Updateable for all measurements
#    Q = DX*X'
#    P = DX*X'
#
#    operator_only && return (K = k, C = C, B = B, Q = Q, P = P)
#
#    return DataDrivenSolution(prob, k, C, B, Q, P, BitVector((true for i in 1:size(X,1))), b, alg, digits = digits, eval_expression = eval_expression)
#end
#
#
#function DiffEqBase.solve(prob::AbstracContProb{dType, false}, alg::AbstractKoopmanAlgorithm;
#    B::AbstractArray = [], digits::Int = 10, operator_only::Bool = false,
#    eval_expression = false,
#    kwargs...) where {dType <: Number}
#    # Check the validity
#    @assert is_valid(prob) "The problem seems to be ill-defined. Please check the problem definition."
#
#    X = prob.X
#    DX = prob.DX
#    U = prob.U
#
#
#    # Create a basis
#    @variables x[1:size(X, 1)] u[1:size(U, 1)] t
#    b = Basis([x; u], x, controls = u, iv = t)
#
#    inds = BitVector([ones(Bool, size(X,1)); zeros(Bool, size(U,1))])
#    C = diagm(ones(dType, size(X,1)))
#
#    # The input maps
#
#    if isempty(B)
#        k, B = alg(X, DX, U)
#    else
#        k, B = alg(X, DX, U, B)
#    end
#    # Updateable for all measurements
#    Q = DX*[X;U]'
#    P = DX*[X;U]'
#
#    operator_only && return (K = k, C = C, B = B, Q = Q, P = P)
#
#    return DataDrivenSolution(prob, k, C, B, Q, P, inds, b, alg, digits = digits, eval_expression = eval_expression)
#end
#
#function DiffEqBase.solve(prob::AbstracContProb{dType, true}, b::Basis, alg::AbstractKoopmanAlgorithm;
#    digits::Int = 10, operator_only::Bool = false,
#    eval_expression = false,
#    kwargs...) where {dType <: Number}
#    # Check the validity
#    @assert is_valid(prob) "The problem seems to be ill-defined. Please check the problem definition."
#
#    X = prob.X
#    DX = prob.DX
#    p = prob.p
#    t = prob.t
#
#    Ψ₀ = b(X, p, t)
#    Ψ₁ = similar(Ψ₀)
#
#    J = jacobian(b)
#
#    for i in 1:size(DX, 2)
#        Ψ₁[:, i] .= J(X[:, i], p, t[i])*DX[:, i]
#    end
#
#    k = alg(Ψ₀, Ψ₁)
#
#    Q = Ψ₁*Ψ₀'
#    P = Ψ₀*Ψ₀'
#    B = zeros(dType, 0, 0)
#
#    # Outpumap -> just the state dependent
#    C = prob.DX / Ψ₁
#
#    operator_only && return (K = k, C = C, B = B, Q = Q, P = P)
#
#    return DataDrivenSolution(prob, k, C, B, Q, P, BitVector((true for i in 1:size(Ψ₀,1))), b, alg, digits = digits, eval_expression = eval_expression)
#end
#
#
#function DiffEqBase.solve(prob::AbstracContProb{dType, false}, b::Basis, alg::AbstractKoopmanAlgorithm;
#    digits::Int = 10, operator_only::Bool = false,
#    eval_expression = false,
#    kwargs...) where {dType <: Number}
#    # Check the validity
#    @assert is_valid(prob) "The problem seems to be ill-defined. Please check the problem definition."
#
#    X = prob.X
#    DX = prob.DX
#    U = prob.U
#    p = prob.p
#    t = prob.t
#
#    Ψ₀ = b(X, p, t, U)
#    Ψ₁ = similar(Ψ₀)
#
#    J = jacobian(b)
#
#    for i in 1:size(DX, 2)
#        Ψ₁[:, i] .= J(X[:, i], p, t[i], U[:,i])*DX[:, i]
#    end
#
#    # Find the indexes of the control states
#    inds = .! is_dependent(map(eq->Num(eq.rhs),equations(b)), Num.(controls(b)))[1,:]
#
#    k, B = alg(Ψ₀[inds, :], Ψ₁[inds, :], Ψ₀[.!inds, :])
#
#    Q = Ψ₁[inds, :]*Ψ₀'
#    P = Ψ₀*Ψ₀'
#
#    # Outpumap -> just the state dependent
#    C = prob.DX / Ψ₁[inds,:]
#
#    operator_only && return (K = k, C = C, B = B, Q = Q, P = P)
#
#    return DataDrivenSolution(prob, k, C, B, Q, P, inds, b, alg, digits = digits, eval_expression = eval_expression)
#end