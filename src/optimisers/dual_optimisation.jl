abstract type AbstractDualOptimiser end;


mutable struct DualOptimiser{S, O} <: AbstractDualOptimiser
    sparse_opt::S
    param_opt::O
end


struct Evaluator
    f::Function
    g::Function
    h::Function
end

function evaluate(c::Evaluator, X::AbstractArray)
    f(p) = c.f(X, p)[1]
    g!(du, p) = c.g(du, X, p)
    h!(du, p) = c.h(du, X, p)
    return f, g!, h!
end


DataDrivenDiffEq.update!(θ::AbstractArray, b::Basis, X::AbstractArray, p::AbstractArray) = θ .= b(X, p = p)

function init_(o::DualOptimiser, X::AbstractArray, A::AbstractArray, Y::AbstractArray, b::Basis)
    # Normal sindy
    Ξ = DataDrivenDiffEq.Optimise.init(o.sparse_opt, A', Y')

    # Generate the cost function and its partial derivatives
    @parameters xi[1:size(Ξ, 2), 1:length(b)]
    # Cost function
    f = simplify_constants(sum((xi*b(X, p = parameters(b))-Y).^2))

    f_oop, f_iip = ModelingToolkit.build_function(f, xi, parameters(b), (), simplified_expr, Val{false})
    g_oop, g_iip = ModelingToolkit.build_function(ModelingToolkit.gradient(f, parameters(b)), xi, parameters(b), (), simplified_expr, Val{false})
    h_oop, h_iip = ModelingToolkit.build_function(ModelingToolkit.hessian(f, parameters(b)), xi, parameters(b), (), simplified_expr, Val{false})

    f_(u, p) = f_oop(u, p)[1]
    g!(du, u, p) = g_iip(du, u, p)
    h!(du, u, p) = h_iip(du, u, p)
    c = Evaluator(f_, g!, h!)

    return Ξ, c
end

function fit_!(Ξ::AbstractArray, p::AbstractArray, X::AbstractArray, A::AbstractArray, Y::AbstractArray, b::Basis, e::Evaluator, opt::DualOptimiser; subiter::Int64 = 1, maxiter::Int64 = 10)

    scales = ones(eltype(Ξ), size(θ, 1))
    for i in 1:maxiter
        # Update θ
        update!(A, b, X, p)
        println(A[1:2, 1:2])
        DataDrivenDiffEq.normalize_theta!(scales, θ)
        # First do a sparsifying regression step
        DataDrivenDiffEq.Optimise.fit!(Ξ, A', Y', opt.sparse_opt, maxiter = subiter)
        DataDrivenDiffEq.rescale_xi!(scales, Ξ)
        # Update the parameter
        res = Optim.optimize(evaluate(e, Ξ)..., p, opt.param_opt, Optim.Options(iterations = subiter))
        p .= res.minimizer
    end

    return res
end
