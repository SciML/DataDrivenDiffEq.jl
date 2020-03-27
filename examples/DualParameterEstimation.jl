using DataDrivenDiffEq
using ModelingToolkit
using OrdinaryDiffEq
using LinearAlgebra
using Plots
using Optim
gr()

# Based upon the paper
# A unified sparse optimization framework to learn
# parsimonious physics-informed models from data
# Champion et.al.
# https://arxiv.org/pdf/1906.10612.pdf

# Algorithm 3 ( with adaptation to be more general )
# General idea:
# Combine Sparse regression with Optim updates for the parameters

function pendulum(u, p, t)
    x = u[2]
    y = -9.81sin(u[1])
    return [x;y]
end

u0 = [0.99π; -1.0]
tspan = (0.0, 20.0)
prob = ODEProblem(pendulum, u0, tspan)
sol = solve(prob, Tsit5(), saveat = 0.3)

plot(sol)

# Create the differential data
DX = similar(sol[:,:])
for (i, xi) in enumerate(eachcol(sol[:,:]))
    DX[:,i] = pendulum(xi, [], 0.0)
end

# Create a basis
@variables u[1:2]
@parameters p[1:2]

# And some other stuff
h = Operation[cos(u[1]+p[1]);u[2]; u[1]*sin(u[2]*p[2])]

basis = Basis(h, u, parameters = p)


abstract type AbstractDualOptimiser end;


mutable struct DualOptimiser{S, P, O, L, B} <: AbstractDualOptimiser
    sparse_opt::S
    param_opt::P
    options::O
    lower::L
    upper::B
end

has_bounds(d::DualOptimiser) = !(isnothing(d.lower) && !isnothing(d.upper))
get_options(d::DualOptimiser) = d.options


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

function init_(X::AbstractArray, A::AbstractArray, Y::AbstractArray, b::Basis; gradient::Bool = true, hessian::Bool = true)

    # Generate the cost function and its partial derivatives
    @parameters xi[1:size(X, 1), 1:length(b)]
    # Cost function
    f = simplify_constants(sum((xi*b(X, p = parameters(b))-Y).^2))

    f_oop, f_iip = ModelingToolkit.build_function(f, xi, parameters(b), (), simplified_expr, Val{false})
    f_(u, p) = f_oop(u, p)[1]

    # TODO how to handle this?
    g!, h! = (args...)->nothing, (args...)->nothing

    if gradient
        g_oop, g_iip = ModelingToolkit.build_function(ModelingToolkit.gradient(f, parameters(b)), xi, parameters(b), (), simplified_expr, Val{false})
        g!(du, u, p) = g_iip(du, u, p)
    end


    if hessian
        h_oop, h_iip = ModelingToolkit.build_function(ModelingToolkit.hessian(f, parameters(b)), xi, parameters(b), (), simplified_expr, Val{false})
        h!(du, u, p) = h_iip(du, u, p)
    end


    c = Evaluator(f_, g!, h!)

    return c
end

function fit_!(Ξ::AbstractArray, p::AbstractArray, X::AbstractArray, A::AbstractArray, Y::AbstractArray, b::Basis, e::Evaluator, opt::DualOptimiser; subiter::Int64 = 1, maxiter::Int64 = 10)

    scales = ones(eltype(Ξ), size(θ, 1))
    for i in 1:maxiter
        # Update θ
        update!(A, b, X, p)
        DataDrivenDiffEq.normalize_theta!(scales, A)
        # First do a sparsifying regression step
        DataDrivenDiffEq.Optimise.fit!(Ξ, A', Y', opt.sparse_opt, maxiter = 1)
        DataDrivenDiffEq.rescale_xi!(scales, Ξ)
        # Update the parameter
        if has_bounds(opt)
            res = Optim.optimize(evaluate(e, Ξ)..., opt.lower, opt.upper, p, opt.param_opt, get_options(opt))
        else
            res = Optim.optimize(evaluate(e, Ξ)..., p, opt.param_opt, get_options(opt))
        end
        #return res
        p .= Optim.minimizer(res)
        #return res
    end

    update!(A, b, X, p)
    DataDrivenDiffEq.normalize_theta!(scales, A)
    # First do a sparsifying regression step
    DataDrivenDiffEq.Optimise.fit!(Ξ, A', Y', opt.sparse_opt, maxiter = 100)
    DataDrivenDiffEq.rescale_xi!(scales, Ξ)

    return
end


# SR3, works good with lesser data and tuning
X = Array(sol)
upper = Float64[π; 1.0]
lower = Float64[-π; -1.0]
options = Optim.Options(iterations = 1)
parameter_optimiser = GradientDescent()
opt = DualOptimiser(SR3(2e-1, 10.0), parameter_optimiser, options, nothing, upper)
has_bounds(opt)
DataDrivenDiffEq.Optimise.get_threshold(opt.sparse_opt)
ps = [0.2; 1.0]
θ = basis(X, p = ps)
E = init_(X, θ, DX, basis, gradient = true, hessian = false) # This takes a long time
Ξ = θ'\DX'
r = fit_!(Ξ, ps, X, θ, DX, basis, E, opt, subiter = 1, maxiter = 2000) #This is super fast
Ψ = Basis(simplify_constants.(Ξ'*basis(variables(basis), p = ps)), u)
println(Ψ)

plot(Ψ(X)')
plot!(DX')
