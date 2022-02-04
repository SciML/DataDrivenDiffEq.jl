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
struct KoopmanSolution{O,IN,S, E, F, A, P}
    k::O
    inds::IN
    sets::S
    error::E
    folds::F
    alg::A
    options::P
end


select_by(::Val, sol::KoopmanSolution) = begin
    @unpack k, error = sol
    i = argmin(error)
    return k[i], error[i]
end

select_by(::Val{:kfold}, sol::KoopmanSolution) = begin
    @unpack k, folds, error  = sol
    size(k, 1) <= 1 && return select_by(1, sol)
    i = argmin(mean(folds, dims = 1)[1,:])
    return k[i], error[i]
end


## Apply the problem to get the operator

# DMD-Like
function CommonSolve.init(prob::AbstractDataDrivenProblem{N,C,P}, alg::AbstractKoopmanAlgorithm, args...; kwargs...) where {N,C,P}
    # Build a basis
    s_x = size(prob.X,1)
    s_u = size(prob.U,1)

    x = [Symbolics.variable(:x, i) for i in 1:s_x]
    u = [Symbolics.variable(:u, i) for i in 1:s_u]
    t = Symbolics.variable(:t)
    
    b = Basis([x; u], x, controls = u, iv = t)
    
    init(prob, b, alg, args...; kwargs...)
end

# All (g(E))DMD like
function CommonSolve.init(prob::AbstractDiscreteProb{N,C}, b::AbstractBasis, alg::A, args...; B = [], eval_expression = false,  kwargs...) where {N,C, A <: AbstractKoopmanAlgorithm}
    @is_applicable prob 

    @unpack X,p,t,U = prob

    x = b(X[:,1:end-1], p, t[1:end-1], U[:,1:end-1])
    y = b(X[:, 2:end], p, t[2:end], U[:, 2:end])
    
    if !isempty(controls(b))
        inds = .! is_dependent(map(eq->Num(eq.rhs),equations(b)), Num.(controls(b)))[1,:]
    else
        inds = ones(Bool, length(b))
    end

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
    return (K, B, C, P, Q,)
end

function operator_error(f, g)
    (x,y,K,B,C,P,Q,inds) -> begin
        k_ = Matrix(K)
        isempty(B) && return g(f(k_*x, C, y))
        return g(f(k_*x[inds, :]+B*x[.! inds, :], C, y))
    end
end

function CommonSolve.solve!(k::KoopmanProblem)
    
    @unpack x, y, b, inds, prob, basis, train, test, alg, options, eval_expression = k
    @unpack normalize, denoise, sampler, maxiter, abstol, reltol, verbose, progress,f,g,digits,kwargs = options
    
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

    sol = KoopmanSolution(ops, inds, (train,test),testerror, trainerror, alg, options)
    
    return DataDrivenSolution(prob, sol, basis, alg; eval_expression = eval_expression, digits = digits, kwargs...)
end
