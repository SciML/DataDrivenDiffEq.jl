
mutable struct ADM{U, O} <: AbstractSubspaceOptimiser
    λ::U
    R::O
end

function ADM(λ::T = 0.1) where T <: Real
    f = NormL1(λ)
    return ADM(λ, f)
end

function fit!(q::AbstractArray{T, 1}, Y::AbstractArray, opt::ADM; maxiter::Int64= 10) where T <: Real
    normalize!(q)
    x = Y*q
    for k in 1:maxiter
        prox!(x, opt.R, Y*q)
        mul!(q, Y', x/norm(Y'*x, 2))
    end
end

function fit!(q::AbstractArray{T, 2}, Y::AbstractArray, opt::ADM; maxiter::Int64= 10) where T <: Real
    @inbounds for i in 1:size(q, 2)
        fit!(view(q, :, i), Y, opt, maxiter = maxiter)
    end
end


# Test

#using LinearAlgebra
#using SparseArrays
#using ProximalOperators

#opt = ADM(1e-2)
#A = Matrix(sprandn(20, 40, 0.1))
#U, S, V = svd(A, full = true)
#
#
#N = nullspace(A, rtol = 0.99)
#X, Y = qr(randn(40,40))
#M = X*N
#rank(M)
#Q = M[:,3]
#norm(Q, 0)
#fit!(Q, N', opt, maxiter = 10000)
#norm(Q, 0)
#fit!(M, N', opt, maxiter = 10000)
