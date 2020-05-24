mutable struct DualOptimiser{B, S, O, P, R} <: AbstractOptimiser
    basis::B # The basis
    sparse_opt::S # Sparse optimiser
    param_opt::O # Optim optimiser
    ps::P # parameters
    bounds::R # bounds
end

"""
    DualOptimiser(basis)
    DualOptimsier(sparse_opt, optim_opt)

`DualOptimiser` allows the sparse regression for a parametrized `Basis`.
"""

DualOptimiser(b) = DualOptimiser(b, STRRidge(), Newton())

function set_threshold!(opt::DualOptimiser, threshold)
    set_threshold!(opt.sparse_opt, threshold)
end

get_threshold(opt::DualOptimiser) = get_threshold(opt.sparse_opt)

init(opt::DualOptimiser, A::AbstractArray, Y::AbstractArray) = init(opt.sparse_opt, A, Y)
init!(X::AbstractArray, opt::DualOptimiser, A::AbstractArray, Y::AbstractArray) = init!(X, opt.sparse_opt, A, Y)

function fit!(X::AbstractArray, A::AbstractArray, Y::AbstractArray, opt::DualOptimiser; maxiter::Int64 = 1, convergence_error::T = eps()) where T <: Real
    println("Fit!")
    return
end

#struct Evaluator
#    f::Function
#    g::Function
#    h::Function
#end
#
#function evaluate(c::Evaluator, X::AbstractArray)
#    f(p) = c.f(X, p)[1]
#    g!(du, p) = c.g(du, X, p)
#    h!(du, p) = c.h(du, X, p)
#    return f, g!, h!
#end
#
#update_theta!(θ::AbstractArray, b::Basis, X::AbstractArray, p::AbstractArray) = θ .= b(X, p = p)
#
#function init_(o::DualOptimiser, X::AbstractArray, A::AbstractArray, Y::AbstractArray, b::Basis)
#    # Normal sindy
#    Ξ = DataDrivenDiffEq.Optimise.init(o.sparse_opt, A', Y')
#
#    # Generate the cost function and its partial derivatives
#    @parameters xi[1:size(Ξ, 2), 1:length(b)]
#    # Cost function
#    f = simplify_constants(sum((xi*b(X, p = parameters(b))-Y).^2))
#
#    # Build an optimisation system
#    opt_sys = OptimizationSystem(f, xi, parameters(b))
#
#    return opt_sys
#end
#
#function fit_!(Ξ::AbstractArray, p::AbstractArray, X::AbstractArray, A::AbstractArray, Y::AbstractArray, b::Basis, e::Evaluator, opt::DualOptimiser; subiter::Int64 = 1, maxiter::Int64 = 10)
#
#    scales = ones(eltype(Ξ), size(θ, 1))
#    for i in 1:maxiter
#        # Update θ
#        update!(A, b, X, p)
#        println(A[1:2, 1:2])
#        DataDrivenDiffEq.normalize_theta!(scales, θ)
#        # First do a sparsifying regression step
#        DataDrivenDiffEq.Optimise.fit!(Ξ, A', Y', opt.sparse_opt, maxiter = subiter)
#        DataDrivenDiffEq.rescale_xi!(scales, Ξ)
#        # Update the parameter
#        res = Optim.optimize(evaluate(e, Ξ)..., p, opt.param_opt, Optim.Options(iterations = subiter))
#        p .= res.minimizer
#    end
#
#    return res
#end
