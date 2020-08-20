mutable struct ADMM{U} <: AbstractOptimizer
    λ::U
    ρ::U
end

"""
    ADMM()
    ADMM(λ, ρ)

`ADMM` is an implementation of Lasso using the alternating direction methods of multipliers and loosely based on [this implementation](https://web.stanford.edu/~boyd/papers/admm/lasso/lasso.html).

`λ` is the sparsification parameter, `ρ` the augmented Lagrangian parameter.

# Example
```julia
opt = ADMM()
opt = ADMM(1e-1, 2.0)
```
"""
ADMM() = ADMM(0.1, 1.0)


function set_threshold!(opt::ADMM, threshold)
    opt.λ = threshold*opt.ρ
end

get_threshold(opt::ADMM) = opt.λ/opt.ρ

init(o::ADMM, A::AbstractArray, Y::AbstractArray) =  A \ Y
init!(X::AbstractArray, o::ADMM, A::AbstractArray, Y::AbstractArray) =  ldiv!(X, qr(A, Val(true)), Y)

#soft_thresholding(x::AbstractArray, t::T) where T <: Real = sign.(x) .* max.(abs.(x) .- t, zero(eltype(x)))

function fit!(X::AbstractArray, A::AbstractArray, Y::AbstractArray, opt::ADMM; maxiter::Int64 = 1, convergence_error::T = eps()) where T <: Real
    n, m = size(A)

    g = NormL1(get_threshold(opt))

    x̂ = deepcopy(X)
    ŷ = zero(X)

    P = I(m)/opt.ρ - (A' * pinv(opt.ρ*I(n) + A*A') *A)/opt.ρ
    c = P*(A'*Y)


    x_i = similar(X)
    x_i .= X

    iters = 0

    _error = zero(eltype(X))
    _sparsity = 0

    @inbounds for i in 1:maxiter
        iters += 1

        x̂ .= P*(opt.ρ*X - ŷ) + c
        prox!(X, g, x̂ + ŷ/opt.ρ)
        ŷ .= ŷ + opt.ρ*(x̂ - X)


        _error = norm(x_i - X, 2)
        _sparsity = norm(X, 0)

        if _error < convergence_error
            break
        else
            x_i .= X
        end

        @logmsg(LogLevel(-1),
            "ADMM",
            _id = :DataDrivenDiffEq,
            message="Error: $(_error)\nSparsity: $(_sparsity)",
            progress=i/maxiter)

    end

    @logmsg(LogLevel(-1),
        "ADMM",
        _id = :DataDrivenDiffEq,
        message="Error: $(_error)\nSparsity: $(_sparsity)",
        progress="done")
    
    X[abs.(X) .< get_threshold(opt)] .= zero(eltype(X))
    return iters
end
