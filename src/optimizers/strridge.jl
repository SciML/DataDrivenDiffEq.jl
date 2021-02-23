# Simple ridge regression based upon the sindy-mpc
# repository, see https://arxiv.org/abs/1711.05501
# and https://github.com/eurika-kaiser/SINDY-MPC/blob/master/LICENSE

mutable struct STRRidge{T} <: AbstractOptimizer
    λ::T
end

"""
    STRRidge(λ = 0.1)

`STRRidge` is taken from the [original paper on SINDY](https://www.pnas.org/content/113/15/3932) and implements a
sequentially thresholded least squares iteration. `λ` is the threshold of the iteration.
It is based upon [this matlab implementation](https://github.com/eurika-kaiser/SINDY-MPC/utils/sparsifyDynamics.m).

# Example
```julia
opt = STRRidge()
opt = STRRidge(1e-1)
```
"""
STRRidge() = STRRidge(0.1)

function set_threshold!(opt::STRRidge, threshold)
    opt.λ = threshold
end

get_threshold(opt::STRRidge) = opt.λ

init(o::STRRidge, A::AbstractArray, Y::AbstractArray) = A \ Y
init!(X::AbstractArray, o::STRRidge, A::AbstractArray, Y::AbstractArray) =  ldiv!(X, qr(A, Val(true)), Y)

function fit!(X::AbstractArray, A::AbstractArray, Y::AbstractArray, opt::STRRidge; maxiter::Int64 = 1, convergence_error::T = eps()) where T <: Real
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

        if norm(x_i - X, 2) < convergence_error
            break
        else
            x_i .= X
        end

    end

    hard_thresholding!(X, get_threshold(opt))
    #X[abs.(X) .< get_threshold(opt)] .= zero(eltype(X))
    return iters
end
