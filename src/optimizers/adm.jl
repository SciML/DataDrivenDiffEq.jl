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
mutable struct ADM{T} <: AbstractSubspaceOptimizer{T}
    """Sparsity threshold"""
    λ::T

    function ADM(threshold = 1e-1)
        @assert all(threshold .> zero(eltype(threshold))) "Threshold must be positive definite"

        return new{typeof(threshold)}(threshold)
    end
end


function (opt::ADM{T})(X, A, Y, λ::V = first(opt.λ);
    maxiter::Int64 = maximum(size(A)), abstol::V = eps(eltype(T)), progress = nothing,
    f::Function = F(opt),
    g::Function = G(opt))  where {T, V}

    n,m = size(A)
    ny, my = size(Y)
    nx, mx = size(X)
    nq, mq = 0,0


    # Closure for the pareto function
    fg(x, A) = (g∘f)(x, A)

    # Init all variables
    R = SoftThreshold()

    xzero = zero(eltype(X))
    obj = xzero
    sparsity = xzero
    conv_measure = xzero

    iters = 0
    converged = false

    max_ind = 0

    nspaces = _assemble_ns(A, Y)

    @inbounds for i in 1:my

        # Add the current data to the regressor
        θ = nspaces[i]'

        N = nullspace(θ', rtol = 0.99)
        Q = deepcopy(N)
        nq, mq = size(Q)

        max_ind = max(max_ind, mq)

        x = N'Q[:, 1]
        q = deepcopy(Q[:, 1])

        for (j,qi) in enumerate(eachcol(Q))
            iters = 0
            converged = false

            while (iters < maxiter) && !converged
                iters += 1

                @views R(x, N'q, λ)
                @views mul!(q, N, x)
                @views normalize!(q, 2)

                if isa(progress, Progress)
                    @views sparsity, obj = f(q,θ)

                    ProgressMeter.next!(
                    progress;
                    showvalues = [
                        (:Threshold, λ), (:Objective, obj), (:Sparsity, sparsity),
                        (:Convergence, conv_measure), (:Measurementcolumn, (i, my)),
                        (:Subspacecolumn, (j, mq))
                        ]
                        )
                end

                conv_measure = norm(q .- qi, 2)

                if conv_measure < abstol
                    converged = true
                else
                    @views q .= qi
                end
            end
         end

        clip_by_threshold!(Q, λ)

        @views for (j, q) in enumerate(eachcol(Q))
            if j == 1
                X[:,i] .= q
            else
                evaluate_pareto!(view(X, :, i), view(q, :), fg, view(θ', :, :))
            end
        end

        if isa(progress, Progress)
            @views sparsity, obj = f(X,θ)

            ProgressMeter.next!(
            progress;
            showvalues = [
                (:Threshold, λ), (:Objective, obj), (:Sparsity, sparsity),
                (:Convergence, conv_measure), (:Column, (i, my))
            ]
            )
        end
    end

    return
end
