
mutable struct ADM{U, O} <: AbstractSubspaceOptimizer
    λ::U
    R::O
end

get_threshold(opt::ADM) = opt.λ
set_threshold!(opt::ADM, λ) = (opt.λ = λ; return)

"""
    ADM(λ = 0.1)

Optimizer for finding a sparse basis vector in a subspace based on [this paper](https://arxiv.org/pdf/1412.4659.pdf).
`λ` is the weight for the soft-thresholding operation.
"""
function ADM(λ::T = 0.1) where T <: Real
    f = NormL1(λ)
    return ADM(λ, f)
end

function fit!(q::AbstractArray{T, 1}, Y::AbstractArray, opt::ADM; maxiter::Int64= 10, tol::T = eps(eltype(q))) where T <: Real
    
    x = Y*q
    q_ = deepcopy(q)
    iters_ = 0
    
    while iters_ <= maxiter
        iters_ += 1
        prox!(x, opt.R, Y*q)
        mul!(q, Y', x)
        normalize!(q, 2)

        if norm(q - q_) < tol 
            break
        else
            q_ .= q
        end
    end

    return iters_
end

function fit!(q::AbstractArray{T, 2}, Y::AbstractArray, opt::ADM; maxiter::Int64= 10, tol::T = eps(eltype(q))) where T <: Real
    iters_ = Inf
    i_ = Inf
    @inbounds for i in 1:size(q, 2)
        i_ = fit!(q[:, i], Y, opt, maxiter = maxiter, tol = tol)
        if iters_ > i_ 
            iters_ = i_
        end
    end

    return iters_
end
