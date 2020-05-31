mutable struct ParallelImplicit{S} <: AbstractSubspaceOptimiser
    opt::S
end

get_threshold(opt::ParallelImplicit) = get_threshold(opt.opt)
set_threshold!(opt::ParallelImplicit, λ) = set_threshold!(opt.opt)

"""
    ParallelImplicit(opt = STRRidge())

Optimiser for finding a sparse basis vector in by try and error of the left hand side as described in [this paper](https://arxiv.org/pdf/1412.4659.pdf).
`opt` is a an `AbstractOptimiser`, e.g. `STRRidge`.
"""

ParallelImplicit(opt = STRRidge()) = ParallelImplicit(STRRidge())


function fit!(X::AbstractArray, A::AbstractArray, Y::AbstractArray, opt::ParallelImplicit; rtol = 0.99, maxiter::Int64 = 1, convergence_error::T = eps(), alg::Optimise.AbstractScalarizationMethod = WeightedSum()) where T <: Real
    # Return just the best candidate for the subspace optimization

    opt_front = ParetoFront(1, scalarization = alg)
    tmp_front = ParetoFront(1, scalarization = alg)

    # New Theta for linear dependent theta
    θ = zeros(eltype(A), size(X, 1), size(A, 2))
    θ[size(A, 1)+1:end, :] .= A

    # We have exactly size
    Q = zeros(eltype(A), size(X, 1), size(X, 1))
    idx = abs(Q[:, 1]) .> zero(eltype(A))

    @inbounds for i in 1:size(Y, 1)
        for j in 1:size(A, 2)
            θ[1:size(A, 1), j] .= Y[i, j]*A[:, j]
        end

        #N = nullspace(θ', rtol = rtol)
        #Q = deepcopy(N) # Deepcopy for inplace

        # Now tryout all columns of Theta as lhs
        for j in 1:size(X, 1)
            idx .= zero(idx)
            idx[j] .= true
            fit!(Q[:, i], view(θ,idx, :)', view(θ, .~ b, :)', opt.opt; maxiter = maxiter, convergence_error = convergence_error)
            fit!(Q, N', opt, maxiter = maxiter)
        end

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
