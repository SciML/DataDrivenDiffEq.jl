using LinearAlgebra
using BenchmarkTools
using Plots

gr()


mutable struct Pareto{X, T}
    ξ::AbstractArray{X,1}
    θ::T
end

mutable struct ParetoFront{X, T, E, L}
    p::AbstractArray{Pareto{X,T},1}
    Ε::E
    Λ::L
end

function RecipesBase.plot(p::ParetoFront; vars = (:Θ, :Ε))
    pl = plot()
    if length(vars) == 2
        scatter(getfield(p, vars[1]), getfield(p, vars[2]), xlabel = String(vars[1]), ylabel = String(vars[2]))
    end
end

function best(p::ParetoFront; objective::F = (x, y, z) -> y) where F <: Function
    vals = map(objective, p.Θ, p.Ε, p.Λ)
    idx = argmin(vals)
    return p.Ξ[:, idx]
end

@inline function softtreshholding(X::AbstractArray, λ::AbstractFloat)
    return sign.(X).* max.(abs.(X) .- λ, 0.0)
end

@inline function ADM(Y::AbstractArray, q_init::AbstractArray; λ::AbstractFloat = 1e-10, maxiter::Int64 = 10, ϵ::AbstractFloat = 1e-3)

    q₁ = q_init
    q₀ = similar(q_init)

    @inbounds for k in 1:maxiter
        q₀ = q₁
        x = softtreshholding(Y*q₁, λ)
        q₁ = Y' * x ./ norm(Y'*x, 2)
        if norm(q₀ - q₁ , 2) <= ϵ
            #println("Found solution with error norm $q_res after $k Iterations.")
            return q₁
        end
    end
    return q₁
end

function ADM!(q₀::AbstractArray, Y::AbstractArray; λ::AbstractFloat = 1e-10, maxiter::Int64 = 10, ϵ::AbstractFloat = 1e-3)
    q₁ = q₀
    @inbounds for k in 1:maxiter
        q₀ = q₁
        x = softtreshholding(Y*q₁, λ)
        q₁ = Y' * x ./ norm(Y'*x, 2)
        if norm(q₀ - q₁ , 2) <= ϵ
            #println("Found solution with error norm $q_res after $k Iterations.")
            return q₁
        end
    end
    return q₁
end

function ADMInitVary(Y::AbstractArray; ϵ::AbstractFloat = 1e-5, λ::AbstractFloat = 5e-2, maxiter::Int64 = 10000)
    # The Y is normalized if given via nullspace
    n,m = size(Y)
    q = zeros(eltype(Y), (m, n))
    Q = zeros(eltype(Y), (n, n))
    n_zeros = zeros(Int64, n)

    @inbounds for i in 1:n
        q₀ = normalize(@view(Y[i, :]))
        q[:, i] = ADMInitVary(Y, q₀; maxiter = maxiter, λ = λ, ϵ = ϵ)
        Q[:, i] = Y*@view(q[:, i])
        n_zeros[i] = length(Q[abs.(Q[:,i]) .< λ, i])
    end

    idx = argmax(n_zeros) # Find sparsest vector
    Ξ = Q[:, idx] # Get the sparsest coefficients
    Ξ[abs.(Ξ) .< λ] .= 0 # Set small coefficients to zeros

    return Pareto(Ξ, sum(abs, abs.(Q[:, idx]) .>= λ))
end

function ADMInitvary!()

function ADMPareto(X::AbstractArray; λ::AbstractFloat = 1e-8, maxiter::Int64 = 100)

    n, m = size(X)
    # Trackable variables
    Ξ = Array{eltype(X)}(undef, m, maxiter)
    Λ = Array{eltype(λ)}(undef, maxiter)
    Ε = Array{eltype(X)}(undef, maxiter)
    θ = Array{Int64}(undef, maxiter)

    Y = nullspace(X)

    @inbounds for i in 1:maxiter
        Λ[i] = (2*(i-1)+1)*λ
        Ξ[:, i], θ[i] = ADM(Y, λ = Λ[i])
        Ε[i] = norm(X*@view(Ξ[:, i]), 2)
    end

    return ParetoFront(Ξ, θ, Ε, Λ)
end

using Profile


X = rand(5, 100)
Y = nullspace(X)

@btime ADM(Y, Y[1,:], λ = 1e-5)
y = Y[1,:]
@btime ADM!(y, Y, λ = 1e-5)

Profile.init()
@btime ADM(Y, λ = 1e-5)

@btime ADMPareto(X, λ = 1e-10, maxiter = 10)

Juno.@profiler ADMPareto(X, λ = 1e-3, maxiter = 1000)

plot(res, vars = (:Λ, :Θ))

getfield(res, :Θ)
test_objective(x, y, z) = abs((1+y)*x)

ξ = best(res, objective = test_objective)
sum(abs.(ξ) .≈ 0)

X*ξ
scatter(Λ, θ, xaxis = :log, yaxis = :log)


Y = nullspace(X)

Ξ, t = ADM(Y, λ = 1e-3, maxiter = 500)
X*Ξ

Q[:, idt]
Array{eltype(X)}(undef, 1000)
Ξ = Q[:, findall(n .> 0.3*maximum(n))]
Ξ[abs.(Ξ) .<= 1e-10] .= 0
rank(Ξ, atol =1e-3)
X*Ξ
Ξ

R = d.R
idx = [sum(abs, ri) >= 1e-5 for ri in eachrow(R)]
R[idx, :]




X*Q
nzeros
Q[abs.(Q) .<= 1e-3] .= 0
Q
