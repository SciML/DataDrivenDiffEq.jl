using DataDrivenDiffEq
using LinearAlgebra
using ModelingToolkit
using OrdinaryDiffEq
using Plots
gr()

function simple_example(du, u, p, t)
    du[1] = -u[2]/(1.0+0.9*sin(u[1]))
    du[2] = -0.9u[2] + 0.1u[1]
end

# Setup the system
u0 = [0.5; 1.0]
p = [0.6 1.5 0.3]
tspan = (0.0, 4.0)
prob = ODEProblem(simple_example, u0, tspan, p)
sol = solve(prob, Tsit5())
plot(sol)

# Collect the derivatives
x = sol[:,:]
dx = similar(x)
for (i, xi) in enumerate(eachcol(x))
    dxi = similar(xi)
    simple_example(dxi, xi, p, 0.0)
    dx[:,i] = dxi
end

# Define the basis
@parameters t
@variables u[1:2] u̇[1:2]
# The dictionary
g = Array{Operation,1}()
push!(g, u[1]^0)
for i in 1:3
    for (ui, dui) in zip(u, u̇)
        push!(g, ui^i)
        push!(g, dui^i)
    end

    for j in i:3
        for (ui,dui) in zip(u, u̇)
            push!(g, ui^i*dui^j)
            push!(g, dui^i*ui^j)
        end
    end
end
push!(g, u̇[1]*sin(u[1]))

basis = Basis(g, [u...; u̇...]);

# Approximate the basis
θ= ISInDy(x, dx, basis, 1e-10, maxiter = 10)
using Lasso
x
X = rand(4, 10000)
Z = zeros(eltype(X), size(basis)[1], size(X)[2])

@inline function evaluate!(y::AbstractArray{T, 1}, basis::Basis, x::AbstractArray{T, 1}) where T <: Number
    y = basis(x)
end



@inline function evaluate!(Z::AbstractArray, X::AbstractArray)
    @inbounds for i in 1:size(X)[2]
        Z[:, i] .= basis(@view X[:, i])
    end
end

@time evaluate!(Z[:, 1], basis, @view X[:, 1])

@time evaluate!(Z, X)

[undef, size(basis), size(X)[2]]
Y = size(hcat(basis.(eachcol(X))...)

nullspace(Y')
Q = qr(nullspace(Y'), Val(true))
R = Q.R
R[abs.(R) .<= 1e-10] .= 0
R

idx = findfirst([sum(oi) for oi in eachcol(R)] .≈ 0)
idx += -1
E = [-inv(R[1:idx-1, 1:idx-1])*R[1:idx-1,idx:end]; Diagonal(ones(size(R)[2]-idx+1))]
Ξ = Matrix(Q.P*E)
scatter([sum(abs2, Y'*xi)+norm(xi, 1) for xi in eachcol(Ξ)])
findmin([sum(abs2, Y'*xi)+norm(xi, 1) for xi in eachcol(Ξ)])
simplify_constants(Ξ[:,31]'*basis.basis)
# Print out the equations
simplify_constants.(Ξ'*basis.basis)
