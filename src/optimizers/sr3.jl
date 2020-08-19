# Based upon alg 1 in
# A unified sparse optimization framework to learn parsimonious physics-informed models from data
# by K Champion et. al.

mutable struct SR3{U,T} <: AbstractOptimizer
    λ::U
    ν::U
    R::T
end


"""
    SR3(λ, ν, R)
    SR3(λ = 1e-1, ν = 1.0)

`SR3` is an optimizer framework introduced [by Zheng et. al., 2018](https://ieeexplore.ieee.org/document/8573778) and used within
[Champion et. al., 2019](https://arxiv.org/abs/1906.10612). `SR3` contains a sparsification parameter `λ`, a relaxation `ν`,
and a corresponding penalty function `R`, which should be taken from [ProximalOperators.jl](https://github.com/kul-forbes/ProximalOperators.jl).

# Examples
```julia
opt = SR3()
opt = SR3(1e-2)
opt = SR3(1e-3, 1.0)
```
"""
function SR3(λ = 1e-1, ν = 1.0)
    R = NormL1
    return SR3(λ, ν, R)
end

function set_threshold!(opt::SR3, threshold)
    opt.λ = threshold^2*opt.ν /2
    return
end

get_threshold(opt::SR3) = sqrt(2*opt.λ/opt.ν)

init(o::SR3, A::AbstractArray, Y::AbstractArray) =  A \ Y
init!(X::AbstractArray, o::SR3, A::AbstractArray, Y::AbstractArray) =  ldiv!(X, qr(A, Val(true)), Y)

function fit!(X::AbstractArray, A::AbstractArray, Y::AbstractArray, opt::SR3; maxiter::Int64 = 1, convergence_error::T = eps(),  progress::B = EmptyProgressMeter()) where {T <: Real, B <: ProgressMeter.AbstractProgress}
    f = opt.R(get_threshold(opt))

    n, m = size(A)
    W = copy(X)

    # Init matrices
    P = inv(A'*A+I(m)/(opt.ν))
    X̂ = A'*Y

    w_i = similar(W)
    w_i .= W
    iters = 0

    _error = zero(eltype(X))
    _sparsity = 0

    for i in 1:maxiter
        iters += 1
        # Solve ridge regression
        X .= P*(X̂+W/(opt.ν))
        # Add proximal iteration
        prox!(W, f, X, opt.ν*opt.λ)

        _error = norm(w_i - W, 2)/opt.ν
        _sparsity = norm(W, 0)

        if _error < convergence_error
            break
        else
            w_i .= W
        end

        next!(progress; showvalues = [(:Iterations, iters),(:Convergence, _error), (:Sparsity, _sparsity)])
        
    end


    finish!(progress; showvalues = [(:Iterations, iters),(:Convergence, _error), (:Sparsity, _sparsity)])
    
    X[abs.(X) .< get_threshold(opt)] .= zero(eltype(X))
    return iters
end
