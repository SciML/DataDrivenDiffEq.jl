"""
$(TYPEDEF)
`ADMM` is an implementation of Lasso using the alternating direction methods of multipliers and loosely based on [this implementation](https://web.stanford.edu/~boyd/papers/admm/lasso/lasso.html).
It solves the following problem
```math
\\min_{x} \\frac{1}{2} \\| Ax-b\\|_2 + \\lambda \\|x\\|_1
```
#Fields
$(FIELDS)
# Example
```julia
opt = ADMM()
opt = ADMM(1e-1, 2.0)
```
"""
mutable struct ADMM{T} <: AbstractOptimizer where T <: Real
    """Sparsity threshold"""
    λ::T
    """Augmented Lagrangian parameter"""
    ρ::T


    function ADMM(threshold = 1e-1, ρ = 1.0)
        @assert threshold > zero(eltype(threshold)) "Threshold must be positive definite"

        return new{typeof(threshold)}(threshold, ρ)
    end
end


function set_threshold!(opt::ADMM, threshold)
    @assert threshold > zero(eltype(threshold)) "Threshold must be positive definite"

    opt.λ = threshold*opt.ρ
end

get_threshold(opt::ADMM) = opt.λ/opt.ρ

init(o::ADMM, A::AbstractArray, Y::AbstractArray) =  A \ Y
init!(X::AbstractArray, o::ADMM, A::AbstractArray, Y::AbstractArray) =  ldiv!(X, qr(A, Val(true)), Y)

function fit!(X::AbstractArray, A::AbstractArray, Y::AbstractArray, opt::ADMM; maxiter::Int64 = 1, convergence_error::T = eps()) where T <: Real
    n, m = size(A)

    x̂ = deepcopy(X)
    ŷ = zero(X)

    P = I(m)/opt.ρ - (A' * pinv(opt.ρ*I(n) + A*A') *A)/opt.ρ
    c = P*(A'*Y)

    R = SoftThreshold()

    x_i = similar(X)
    x_i .= X

    iters = 0

    @views for i in 1:maxiter
        iters += 1

        x̂ .= P*(opt.ρ.*X .- ŷ) .+ c
        R(X,  x̂ .+ ŷ./opt.ρ, get_threshold(opt))
        ŷ .= ŷ .+ opt.ρ.*(x̂ .- X)

        if norm(x_i .- X, 2) < convergence_error
            break
        else
            x_i .= X
        end

    end


    clip_by_threshold!(X, get_threshold(opt))
    return iters
end
