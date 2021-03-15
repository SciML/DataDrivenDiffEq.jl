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
```
## Note
This was formally `STRRidge` and has been renamed.
"""
mutable struct STLSQ{T} <: AbstractOptimizer where T <: Real
    """Sparsity threshold"""
    λ::T

    function STLSQ(threshold = 1e-1)
        @assert threshold > zero(eltype(threshold)) "Threshold must be positive definite"

        return new{typeof(threshold)}(threshold)
    end
end


function set_threshold!(opt::STLSQ, threshold)
    @assert threshold > zero(eltype(threshold)) "Threshold must be positive definite"
    opt.λ = threshold
end

get_threshold(opt::STLSQ) = opt.λ

init(o::STLSQ, A::AbstractArray, Y::AbstractArray) = A \ Y
init!(X::AbstractArray, o::STLSQ, A::AbstractArray, Y::AbstractArray) =  ldiv!(X, qr(A, Val(true)), Y)

function fit!(X::AbstractArray, A::AbstractArray, Y::AbstractArray, opt::STLSQ; maxiter::Int64 = 1, convergence_error::T = eps()) where T <: Real

    smallinds = abs.(X) .<= opt.λ
    biginds = @. ! smallinds[:, 1]

    x_i = similar(X)
    x_i .= X

    iters = 0

    for i in 1:maxiter
        iters += 1

        smallinds .= abs.(X) .<= opt.λ
        X[smallinds] .= zero(eltype(X))
        @views for j in 1:size(Y, 2)
            @. biginds = ! smallinds[:, j]
            X[biginds, j] .= A[:, biginds] \ Y[:,j]
        end

        if norm(x_i .- X, 2) < convergence_error
            break
        else
            x_i .= X
        end

    end

    clip_by_threshold!(X, get_threshold(opt))
    return iters
end
