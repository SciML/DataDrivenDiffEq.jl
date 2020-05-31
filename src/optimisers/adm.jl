
mutable struct ADM{U, O} <: AbstractSubspaceOptimiser
    λ::U
    R::O
end

get_threshold(opt::ADM) = opt.λ
set_threshold!(opt::ADM, λ) = (opt.λ = λ; return)

"""
    ADM(λ = 0.1)

Optimiser for finding a sparse basis vector in a subspace based on [this paper](https://arxiv.org/pdf/1412.4659.pdf).
`λ` is the weight for the soft-thresholding operation.
"""
function ADM(λ::T = 0.1) where T <: Real
    f = NormL1(λ)
    return ADM(λ, f)
end

# Inner Nullspace optimiser
function fit!(q::AbstractArray{T, 1}, Y::AbstractArray, opt::ADM; maxiter::Int64= 10) where T <: Real
    normalize!(q)
    x = Y*q
    for k in 1:maxiter
        prox!(x, opt.R, Y*q)
        mul!(q, Y', x/norm(Y'*x, 2))
    end

    q[abs.(q) .< get_threshold(opt)] .= zero(eltype(q))
    normalize!(q, 2)
    return
end

# Inner Nullspace optimiser
function fit!(q::AbstractArray{T, 2}, Y::AbstractArray, opt::ADM; maxiter::Int64= 10) where T <: Real
    @inbounds for i in 1:size(q, 2)
        fit!(view(q, :, i), Y, opt, maxiter = maxiter)
    end

    return
end

function fit!(X::AbstractArray, A::AbstractArray, Y::AbstractArray, opt::ADM; rtol = 0.99, maxiter::Int64 = 1, convergence_error::T = eps(), alg::Optimise.AbstractScalarizationMethod = WeightedSum()) where T <: Real
    # Return just the best candidate for the subspace optimization

    opt_front = ParetoFront(1, scalarization = alg)
    tmp_front = ParetoFront(1, scalarization = alg)

    θ = zeros(eltype(A), size(X, 1), size(A, 2))
    θ[size(A, 1)+1:end, :] .= A


    @inbounds for i in 1:size(Y, 1)
        for j in 1:size(A, 2)
            θ[1:size(A, 1), j] .= Y[i, j]*A[:, j]
        end

        N = nullspace(θ', rtol = rtol)
        Q = deepcopy(N) # Deepcopy for inplace

        # Find sparse vectors in nullspace
        # Calls effectively the ADM algorithm with varying initial conditions
        fit!(Q, N', opt, maxiter = maxiter)

        # Compute pareto front
        @inbounds for (i, qi) in enumerate(eachcol(Q))
            if i == 1
                set_candidate!(opt_front, 1, [norm(qi, 0); norm(θ'*qi, 2)], qi, maxiter, get_threshold(opt))
                set_candidate!(tmp_front, 1, [norm(qi, 0); norm(θ'*qi, 2)], qi, maxiter, get_threshold(opt))
            else
                set_candidate!(tmp_front, 1, [norm(qi, 0); norm(θ'*qi, 2)], qi, maxiter, get_threshold(opt))
                conditional_add!(opt_front, tmp_front)
            end
        end

        #println(parameter(opt_front[1]))
        X[:, i] .= parameter(opt_front[1])
        X[:, i] .= X[:, i] ./ maximum(abs.(X[:, i]))
    end
    return
end
