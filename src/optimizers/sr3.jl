#Based upon alg 2 in https://ieeexplore.ieee.org/document/8573778

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
mutable struct SR3{T, V, P <: AbstractProximalOperator} <: AbstractOptimizer{T}
   """Sparsity threshold"""
   λ::T
   """Relaxation parameter"""
   ν::V
   """Proximal operator"""
   R::P

   function SR3(threshold::T = 1e-1, ν::V = 1.0, R::P = HardThreshold()) where {T,V,P}
      @assert all(threshold .> zero(eltype(threshold))) "Threshold must be positive definite"
      @assert ν > zero(V) "Relaxation must be positive definite"

       λ = isa(R, HardThreshold) ? threshold.^2 /2 : threshold
       return new{T, V, P}(λ, ν, R)
   end
end

function (opt::SR3{T,V,R})(X, A, Y, λ::V = first(opt.λ);
    maxiter::Int64 = maximum(size(A)), abstol::V = eps(eltype(T)), progress = nothing)  where {T, V, R}

   n, m = size(A)
   ν = opt.ν
   W = copy(X)

   # Init matrices
   H = cholesky(A'*A+I(m)*opt.ν)
   X̂ = A'*Y

   w_i = similar(W)
   @views w_i .= W
   iters = 0

   iters = 0
   converged = false

   xzero = zero(eltype(X))
   obj = xzero
   sparsity = xzero
   conv_measure = xzero

   while (iters < maxiter) && !converged
       iters += 1

       # Solve ridge regression
       @views ldiv!(X, H, X̂ .+ W*ν)
       #X .= H*(X̂ .+ W*opt.ν)
       # Proximal
       @views opt.R(W, X, λ)

       @views conv_measure = norm(w_i .- W, 2)

       if isa(progress, Progress)
           @views obj = norm(Y .- A*W, 2)
           @views sparsity = norm(W, 0)

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
       else
           @views w_i .= W
       end
   end
   # We really search for W here
   @views X .= W
   @views clip_by_threshold!(X, λ)
   return iters
end
