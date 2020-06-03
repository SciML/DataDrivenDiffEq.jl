mutable struct ParallelImplicit{S} <: AbstractSubspaceOptimiser
    opt::S
end

get_threshold(opt::ParallelImplicit) = get_threshold(opt.opt)
set_threshold!(opt::ParallelImplicit, λ) = set_threshold!(opt.opt)

"""
    ParallelImplicit()

Optimiser for finding a sparse basis vector in by try and error of the left hand side as described in [this paper](https://arxiv.org/pdf/1412.4659.pdf).
`opt` is a an `AbstractOptimiser`, e.g. `STRRidge`.
"""

ParallelImplicit() = ParallelImplicit(STRRidge())


function fit!(X::AbstractArray, A::AbstractArray, Y::AbstractArray, opt::ParallelImplicit; rtol = 0.99, maxiter::Int64 = 1, convergence_error::T = eps(), alg::Optimise.AbstractScalarizationMethod = WeightedSum()) where T <: Real
    # Return just the best candidate for the subspace optimization

    # New Theta for linear dependent theta
    θ = zeros(eltype(A), size(X, 1), size(A, 2))
    θ[size(A, 1)+1:end, :] .= A

    # TODO maybe add normalization here
    @inbounds for i in 1:size(Y, 1)
        for j in 1:size(A, 2)
            θ[1:size(A, 1), j] .= Y[i, j]*A[:, j]
        end

        X[:, i] = fit!(θ, opt, maxiter = maxiter, convergence_error = convergence_error, alg = alg)
    end
    X[abs.(X) .< get_threshold(opt.opt)] .= zero(eltype(X))
    return
end

function fit!(A::AbstractArray, opt::ParallelImplicit; maxiter::Int64 = 1, convergence_error::T = eps(), alg::Optimise.AbstractScalarizationMethod = WeightedSum()) where T <: Real
    # Solve the optimisation problem for a single candidate library
    opt_front = ParetoFront(1, scalarization = alg)
    tmp_front = ParetoFront(1, scalarization = alg)

    # We have exactly size
    qi = init(opt.opt, A', A[1, :])
    idx = trues(length(qi))

    # Now tryout all columns of Theta as lhs
    for j in 1:size(A, 1)
        qi .= zero(eltype(qi))
        qi[j] = one(eltype(qi))
        idx .= true
        idx[j] = false

        qi[idx] = init(opt.opt, A[idx, :]', -A[j, :])
        iters = fit!(qi[idx], A[idx, :]', -A[j, :], opt.opt,  maxiter = maxiter, convergence_error = convergence_error)

        if j == 1
            set_candidate!(opt_front, 1, [norm(qi[idx], 0); norm(A[j,:] - A[idx, :]'*qi[idx], 2)], deepcopy(qi), iters, get_threshold(opt))
            set_candidate!(tmp_front, 1, [norm(qi[idx], 0); norm(A[j,:] - A[idx, :]'*qi[idx], 2)], deepcopy(qi), iters, get_threshold(opt))
        else
            set_candidate!(tmp_front, 1, [norm(qi[idx], 0); norm(A[j,:] - A[idx, :]'*qi[idx], 2)], deepcopy(qi), iters, get_threshold(opt))
            conditional_add!(opt_front, tmp_front)
        end
    end
    qi = parameter(opt_front[1])
    qi = qi ./ maximum(abs.(qi))
    return qi
end
