"""
$(TYPEDEF)
Optimizer for finding a sparse basis vector in a subspace based on [this paper](https://arxiv.org/pdf/1412.4659.pdf).
It solves the following problem
```math
\\min_{x} \\|x\\|_0 ~s.t.~Ax= 0
```
#Fields
$(FIELDS)
# Example
```julia
ADM()
ADM(λ = 0.1)
```
"""
mutable struct ADM{U} <: AbstractSubspaceOptimizer
    """Sparsity threshold"""
    λ::U

    function ADM(threshold = 1e-1)
        @assert threshold > zero(eltype(threshold)) "Threshold must be positive definite"

        return new{typeof(threshold)}(threshold)
    end
end

get_threshold(opt::ADM) = opt.λ
function set_threshold!(opt::ADM, threshold)
    @assert threshold > zero(eltype(threshold)) "Threshold must be positive definite"

    opt.λ = threshold
    return
end


# ADM algorithm
function fit!(q::AbstractArray{T, 1}, Y::AbstractArray, opt::ADM; maxiter::Int64= 10, tol::T = eps(eltype(q))) where T <: Real

    x = Y*q
    q_ = deepcopy(q)
    iters_ = 0

    R = SoftThreshold()

    while iters_ <= maxiter
        iters_ += 1
        R(x, Y*q, get_threshold(opt))
        #prox!(x, opt.R, Y*q)
        mul!(q, Y', x)
        normalize!(q, 2)

        if norm(q - q_) < tol
            break
        else
            q_ .= q
        end
    end

    return iters_
end

# ADM initvary
function fit!(q::AbstractArray{T, 2}, Y::AbstractArray, opt::ADM; maxiter::Int64= 10, tol::T = eps(eltype(q))) where T <: Real
    iters_ = Inf
    i_ = Inf

    @inbounds for i in 1:size(q, 2)
        i_ = fit!(q[:, i], Y, opt, maxiter = maxiter, tol = tol)
        if iters_ > i_
            iters_ = i_
        end
    end

    return iters_
end

# ADM pareto
function fit!(X::AbstractArray, A::AbstractArray, Y::AbstractArray, opt::ADM; rtol = 0.99, maxiter::Int64 = 1, convergence_error::T = eps(), f::Function = (xi, theta)->[norm(xi, 0); norm(theta'*xi, 2)], g::Function = x->norm(x),) where T <: Real
    # Return just the best candidate for the subspace optimization

    θ = zeros(eltype(A), size(X, 1), size(A, 2))
    θ[size(A, 1)+1:end, :] .= A


    fg(xi, theta) = (g∘f)(xi, theta)

    iters = Inf

    @inbounds for i in 1:size(Y, 1)
        for j in 1:size(A, 2)
            @views θ[1:size(A, 1), j] .= Y[i, j].*A[:, j]
        end

        N = nullspace(θ', rtol = rtol)
        Q = deepcopy(N) # Deepcopy for inplace
        # Find sparse vectors in nullspace
        # Calls effectively the ADM algorithm with varying initial conditions
        fit!(Q, N', opt, maxiter = maxiter)

        # Find sparse vectors in nullspace
        # Calls effectively the ADM algorithm with varying initial conditions
        iters_ = fit!(Q, N', opt, maxiter = maxiter, tol = convergence_error)
        iters_ < iters ? iters = iters_ : nothing

        # Compute pareto front
        for (j, ξ) in enumerate(eachcol(Q))
            if j == 1
                X[:, i] .= ξ
            else
                evaluate_pareto!(view(X, :, i), view(ξ, :), fg, view(θ, :, :))
            end
        end

        X[abs.(X[:, i]) .< get_threshold(opt), i] .= zero(eltype(X))
        @views X[:, i] .= X[:, i] ./ maximum(abs, X[:, i])
    end


    return iters
end
