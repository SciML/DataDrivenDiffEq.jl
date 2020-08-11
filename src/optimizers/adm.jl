
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

function fit!(q::AbstractArray{T, 1}, Y::AbstractArray, opt::ADM; maxiter::Int64= 10) where T <: Real
    
    x = Y*q
    
    for k in 1:maxiter
        prox!(x, opt.R, Y*q)
        mul!(q, Y', x/norm(Y'*x, 2))
    end

    return
end

function fit!(q::AbstractArray{T, 2}, Y::AbstractArray, opt::ADM; maxiter::Int64= 10) where T <: Real
    @inbounds for i in 1:size(q, 2)
        fit!(view(q, :, i), view(Y, :, :), opt, maxiter = maxiter)
    end

    return
end
