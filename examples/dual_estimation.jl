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
    y = -9.81sin(u[1]) + 0.5*sin(t)
    return [x;y]
end

u0 = [0.99π; -1.0]
tspan = (0.0, 5.0)
prob = ODEProblem(pendulum, u0, tspan)
sol = solve(prob, Tsit5(), saveat = 0.3)

plot(sol)

# Create the differential data
DX = similar(sol[:,:])
for (i, xi) in enumerate(eachcol(sol[:,:]))
    DX[:,i] = pendulum(xi, [], 0.0)
end

# Create a basis
@variables u[1:2] t
@parameters p[1:3]

# And some other stuff
h = Operation[cos(u[1]+p[1]); t;u[2]; u[1] ;u[1]*sin(u[2]*p[2]); sin(p[3]*t)]

basis = Basis(h, u, parameters = p, iv = t)


# SR3, works good with lesser data and tuning
X = Array(sol)
upper = Float64[π; 10.0; 10.0]
lower = Float64[-π; -10.0; -10.0]
parameter_optimiser = Newton()
ps = randn(3)
opt = DataDrivenDiffEq.Optimise.DualOptimiser(basis, STRRidge(), Newton(), ps, (lower, upper))
θ = basis(X, ps, sol.t)
Ξ = DataDrivenDiffEq.Optimise.init(opt, θ', DX')

struct Evaluator{F, G, H}
    f::F
    g::G
    h::H
end

function (c::Evaluator)(Ξ, X , DX, t)
    function fgh!(F, G, H, x)
        G == nothing || c.g(G, x, Ξ, X, DX, t)
        H == nothing || c.h(H, x, Ξ, X, DX, t)
        F == nothing || return c.f(p, Ξ, X, DX, t)
        nothing
    end

    return Optim.only_fgh!(fgh!)
end

function generate_evaluator(X::AbstractArray, DX::AbstractArray, t, opt::DataDrivenDiffEq.Optimise.DualOptimiser)
    # Generate the cost function and its partial derivatives
    @variables xi[1:size(DX, 1), 1:length(opt.basis)]
    @variables dx_[1:size(DX, 1)]
    # Cost function
    f_t = sum(abs2, xi*opt.basis.basis-dx_)
    osys = OptimizationSystem(f_t, parameters(basis), [[xi...]; variables(opt.basis); dx_; independent_variable(opt.basis)])
    f_ = generate_function(osys,  expression = Val{false})
    g_ = generate_gradient(osys, expression = Val{false})[1]
	h_ = generate_hessian(osys, expression = Val{false})[1]

	# Create closure 
	function f(p, Ξ, X, DX, t)
		cost = zero(eltype(X))
		@inbounds for i in 1:size(X, 2)
			cost += f_(p, [Ξ...; X[:, i]; DX[:, i]; t[i]])
		end
		return cost
	end

	function g!(g, p, Ξ, X, DX, t)
		g .= zero(g)
		@inbounds for i in 1:size(X, 2)
			g .+= g_(p, [Ξ...; X[:, i]; DX[:, i]; t[i]])
		end
		return nothing
	end

	function h!(h, p, Ξ, X, DX, t)
		h .= zero(h)
		@inbounds for i in 1:size(X, 2)
			h .+= h_(p, [Ξ...; X[:, i]; DX[:, i]; t[i]])
		end
		return nothing
	end

    c = Evaluator(f, g!, h!)
end


function update_theta!(Θ, X, p, t, opt::DataDrivenDiffEq.Optimise.DualOptimiser)
	opt.basis(Θ, X, p, t)
end


update_theta!(θ, X, randn(3), sol.t, opt)
θ
osys = generate_evaluator(X[:, :], DX[1:2, :], sol.t[:], opt)


osys(Ξ', X, DX, sol.t)

function sparse_reg(X::AbstractArray, Ẋ::AbstractArray, Ψ::Basis, p::AbstractArray, t::AbstractVector , maxiter::Int64 , opt::DataDrivenDiffEq.Optimise.DualOptimiser, denoise::Bool, normalize::Bool, convergence_error)
    @assert size(X)[end] == size(Ẋ)[end]
    nx, nm = size(X)
    ny, nm = size(Ẋ)

    Ξ = zeros(eltype(X), length(Ψ), ny)
    scales = ones(eltype(X), length(Ψ))
    θ = Ψ(X, p, t)

    denoise ? optimal_shrinkage!(θ') : nothing
    normalize ? DataDrivenDiffEq.normalize_theta!(scales, θ) : nothing

	DataDrivenDiffEq.Optimise.init!(Ξ, opt, θ', Ẋ')

	osys= generate_evaluator(X, Ẋ, t, opt)

	for i in 1:maxiter
		_ = DataDrivenDiffEq.Optimise.fit!(Ξ, θ', Ẋ', opt.sparse_opt, maxiter = 1, convergence_error = convergence_error)
		res = Optim.optimize(osys(Ξ', X, Ẋ, t), opt.ps, opt.param_opt, Optim.Options(iterations = 1))
		update_theta!(θ, X, opt.ps, t)
		normalize ? normalize_theta!(scales, θ) : nothing
	end

    normalize ? DataDrivenDiffEq.rescale_xi!(Ξ, scales) : nothing

    return Ξ, iters
end


sparse_reg(X, DX, basis, ps, sol.t, 2, opt, false, false, eps())

function fit!(Ξ::AbstractArray, X::AbstractArray, A::AbstractArray, Y::AbstractArray, t::AbstractArray, opt::DataDrivenDiffEq.Optimise.DualOptimiser; subiter::Int64 = 1, maxiter::Int64 = 10)
    osys= generate_evaluator(X[:, 1:5], DX[1:1, 1:5], sol.t[1:5], opt)

    scales = ones(eltype(Ξ), size(A, 1))
    for i in 1:maxiter
        # Update θ
		update_theta!(A, X, opt.ps, t, opt)
		Ξ, iters = iters = sparse_regression!(Ξ, A, Y, 1, opt.sparse_opt, false, false, convergence_error)
		
        # First do a sparsifying regression step
        DataDrivenDiffEq.Optimise.fit!(Ξ, A', Y', opt.sparse_opt, maxiter = subiter)
        DataDrivenDiffEq.rescale_xi!(scales, A)
        # Update the parameter
        res = Optim.optimize(osys(A, X, DX, t), opt.ps, opt.param_opt, Optim.Options(iterations = subiter))
        opt.ps .= res.minimizer
    end

    return opt.ps
end

fit!(Ξ, X, θ, DX, sol.t, opt)


g_oop, g! = generate_gradient(o_sys, parameters(basis))''

f_oop, f_iip = ModelingToolkit.build_function(bs, vs, ps, [iv], expression = Val{false})

fff = sum(u)

parameters(basis)

build_function(fff, u, [], [])

equations(o_sys)
states(o_sys)
parameters(o_sys)

calculate_gradient(o_sys)
calculate_hessian(combinedsys)
generate_function(combinedsys)
generate_gradient(combinedsys)
generate_hessian(combinedsys)

@time simplify(sum(f_target))
ps = [-10.0; 0.3; 0.2]
θ = basis(X, p = ps)
E = init_(X, θ, DX, basis, gradient = true, hessian = true) # This takes a long time
Ξ = qr(θ')\DX'
res = fit_!(Ξ, ps, X, θ, DX, basis, E, opt, maxiter = 2000) #This is super fast
ps
Ψ = Basis(simplify_constants.(Ξ'*basis(variables(basis), p = ps)), u)
println(Ψ)
plot(Ψ(X)')
plot!(DX')
