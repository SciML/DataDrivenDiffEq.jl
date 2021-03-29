# Simple ridge regression based upon the sindy-mpc
# repository, see https://arxiv.org/abs/1711.05501
# and https://github.com/eurika-kaiser/SINDY-MPC/blob/master/LICENSE

"""
$(TYPEDEF)
`STLQS` is taken from the [original paper on SINDY](https://www.pnas.org/content/113/15/3932) and implements a
sequentially thresholded least squares iteration. `λ` is the threshold of the iteration.
It is based upon [this matlab implementation](https://github.com/eurika-kaiser/SINDY-MPC/utils/sparsifyDynamics.m).
It solves the following problem
```math
\\min_{x} \\frac{1}{2} \\| Ax-b\\|_2 + \\lambda \\|x\\|_2
```
#Fields
$(FIELDS)
# Example
```julia
opt = STLQS()
opt = STLQS(1e-1)
opt = STLQS(Float32[1e-2; 1e-1])
```
## Note
This was formally `STRRidge` and has been renamed.
"""
mutable struct STLSQ{T} <: AbstractOptimizer{T}
    """Sparsity threshold"""
    λ::T

    function STLSQ(threshold::T = 1e-1) where T
        @assert all(threshold .> zero(eltype(threshold))) "Threshold must be positive definite"

        return new{typeof(threshold)}(threshold)
    end

end

Base.summary(::STLSQ) = "STLSQ"

function (opt::STLSQ{T})(X, A, Y, λ::U = first(opt.λ);
    maxiter = maximum(size(A)), abstol::U = eps(eltype(T)),
    progress = nothing) where {T,U}

    smallinds = abs.(X) .<= λ
    biginds = @. ! smallinds[:, 1]

    x_i = similar(X)
    x_i .= X

    xzero = zero(eltype(X))
    obj = xzero
    sparsity = xzero
    conv_measure = xzero

    iters = 0
    converged = false

    _progress = isa(progress, Progress)
    initial_prog = _progress ? progress.counter : 0


    while (iters < maxiter) && !converged
        iters += 1

        smallinds .= abs.(X) .<= λ
        X[smallinds] .= xzero

        for j in 1:size(Y, 2)
            @. biginds = ! smallinds[:, j]
            X[biginds, j] .= A[:, biginds] \ Y[:,j]
        end

        conv_measure = norm(x_i .- X, 2)

        if _progress
            obj = norm(Y - A*X, 2)
            sparsity = norm(X, 0, λ)

            ProgressMeter.next!(
            progress;
            showvalues = [
                (:Threshold, λ), (:Objective, obj), (:Sparsity, sparsity),
                (:Convergence, conv_measure)
            ]
            )
        end

        if conv_measure < abstol
            converged = true

            _progress ? (progress.counter = initial_prog + maxiter) : nothing

            #    ProgressMeter.update!(
            #    progress,
            #    initial_prog + maxiter
            #    )
            #end


        else
            x_i .= X
        end
    end

    clip_by_threshold!(X, λ)
    return
end
