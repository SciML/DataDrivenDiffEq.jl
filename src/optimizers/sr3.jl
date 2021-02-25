# Based upon alg 2 in https://ieeexplore.ieee.org/document/8573778

"""
$(TYPEDEF)

`SR3` is an optimizer framework introduced [by Zheng et. al., 2018](https://ieeexplore.ieee.org/document/8573778) and used within
[Champion et. al., 2019](https://arxiv.org/abs/1906.10612). `SR3` contains a sparsification parameter `λ`, a relaxation `ν`.

It solves the following problem

```math
\\min_{x, w} \\frac{1}{2} \\| Ax-b\\|_2 + \\lambda R(w) + \\frac{\\nu}{2}\\|x-w\\|_2
```

Where `R` is a proximal operator and the result is given by `w`.

#Fields
$(FIELDS)

# Example
```julia
opt = SR3()
opt = SR3(1e-2)
opt = SR3(1e-3, 1.0)
opt = SR3(1e-3, 1.0, SoftThreshold())
```

## Note
Opposed to the original formulation, we use `ν` as a relaxation parameter,
as given in [Champion et. al., 2019](https://arxiv.org/abs/1906.10612). In the standard case of
hard thresholding the sparsity is interpreted as `λ = threshold^2 / 2`, otherwise `λ = threshold`.

"""
mutable struct SR3{T,P} <: AbstractOptimizer where {T <: Real, P <: AbstractProximalOperator}
    """Sparsity threshold"""
    λ::T
    """Relaxation parameter"""
    ν::T
    """Proximal operator"""
    R::P

    function SR3(threshold = 1e-1, ν = 1.0, R = HardThreshold())
        @assert threshold > zero(eltype(threshold)) "Threshold must be positive definite"
        @assert ν > zero(eltype(threshold)) "Relaxation must be positive definite"

        λ = isa(R, HardThreshold) ? threshold^2 /2 : threshold
        return new{typeof(λ), typeof(R)}(λ, ν, R)
    end
end

function set_threshold!(opt::SR3, threshold)
    @assert threshold > zero(eltype(threshold)) "Threshold must be positive definite"

    opt.λ =  isa(opt.R, HardThreshold) ? threshold^2 /2 : threshold
    return
end

get_threshold(opt::SR3) = opt.λ/opt.ν

init(o::SR3, A::AbstractArray, Y::AbstractArray) =  A \ Y
init!(X::AbstractArray, o::SR3, A::AbstractArray, Y::AbstractArray) =  ldiv!(X, qr(A, Val(true)), Y)

function fit!(X::AbstractArray, A::AbstractArray, Y::AbstractArray, opt::SR3; maxiter::Int64 = 1, convergence_error::T = eps()) where T <: Real

    n, m = size(A)
    W = copy(X)

    # Init matrices
    H = cholesky(A'*A+I(m)*opt.ν)
    X̂ = A'*Y

    w_i = similar(W)
    w_i .= W
    iters = 0

    for i in 1:maxiter
        iters += 1

        # Solve ridge regression
        @views ldiv!(X, H, X̂ .+ W*opt.ν)
        #X .= H*(X̂ .+ W*opt.ν)
        # Proximal
        @views opt.R(W, X, get_threshold(opt))

        if norm(w_i .- W, 2)*opt.ν < convergence_error
            break
        else
            w_i .= W
        end

    end
    # We really search for W here
    X .= W
    #clip_by_threshold!(X, get_threshold(opt))
    return iters
end
