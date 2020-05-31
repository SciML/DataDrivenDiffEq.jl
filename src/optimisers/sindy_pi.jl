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

    opt_front = ParetoFront(1, scalarization = alg)
    tmp_front = ParetoFront(1, scalarization = alg)

    # New Theta for linear dependent theta
    θ = zeros(eltype(A), size(X, 1), size(A, 2))
    θ[size(A, 1)+1:end, :] .= A

    # We have exactly size
    qi = zeros(eltype(A), size(X, 1), 1)
    idx = abs.(qi) .> zero(eltype(A))
    n_idx =  @. ! idx

    @inbounds for i in 1:size(Y, 1)
        for j in 1:size(A, 2)
            θ[1:size(A, 1), j] .= Y[i, j]*A[:, j]
        end

        # Now tryout all columns of Theta as lhs
        for j in 1:size(X, 1)
            idx .= zero(idx)
            idx[j] = true
            n_idx .= @. ! idx
            qi[j, 1] = one(eltype(qi))
            qi[n_idx, :] .= init(opt.opt, θ[n_idx, :]', θ[idx, :]')
            iters = fit!(qi[n_idx, :], θ[n_idx, :]', θ[idx, :]', opt.opt,  maxiter = maxiter, convergence_error = convergence_error)
            if i == 1
                set_candidate!(opt_front, 1, [norm(qi, 0); norm(θ'*qi, 2)], qi, iters, get_threshold(opt))
                set_candidate!(tmp_front, 1, [norm(qi, 0); norm(θ'*qi, 2)], qi, iters, get_threshold(opt))
            else
                set_candidate!(tmp_front, 1, [norm(qi, 0); norm(θ'*qi, 2)], qi, iters, get_threshold(opt))
                conditional_add!(opt_front, tmp_front)
            end
        end

        X[:, i] .= parameter(opt_front[1])[:, 1]
    end
    return
end
