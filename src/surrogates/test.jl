using LinearAlgebra
using ModelingToolkit
using ForwardDiff
using DataDrivenDiffEq
using AbstractDifferentiation

import DataDrivenDiffEq: AbstractDataDrivenProblem, DirectDataDrivenProblem

using Statistics
using StatsBase
using Plots
include("types.jl")
include("investigate.jl")
include("derivative.jl")

f(x) = [exp10(first(x)-last(x)); x[1]*x[3]+x[1]*x[2]; sum(x); prod(x[1:2])/x[3]] 
f(x::AbstractMatrix) = hcat(map(f, eachcol(x))...)
x = rand(3, 1000)
y = f(x) 



y .+= 1e-3*randn(size(y))

s = DataDrivenSurrogate(f, x, opts)
println(s[4])

## Convert to equations
@variables z[1:3]
z = collect(z)

println(s[2])

s[2].left(z)
simplify(s[2].right(z))

norm(y[2:2, :] - s[2](x)) 

mean(x[2,:])






function flatten(x::AbstractSurrogate)
    return x
end

function flatten(x::CompositeSurrogate)
    return reduce(vcat, map(flatten , [x.right, x.left]))
end

surrs[3].t


s_ = flatten(surrs[4])

opt = ImplicitOptimizer()
bs = Basis(polynomial_basis(z, 2), z)
res = map(s_) do si
    ŷ = si(x)
    prob = DirectDataDrivenProblem(x,ŷ)
    _r = nothing
    if typeof(si) <: LinearSurrogate
        solve(prob, Basis(polynomial_basis(z, 1), z), STLSQ())
    elseif typeof(si) <: NonlinearSurrogate
        #solve(prob, bs, opt)
    else 
        nothing
    end
end



println.(res[1:2], Val{true})

# Create for each of the surrogates a SINDy problem and collect the results


@variables z[1:3]
z = collect(z)




p = DirectDataDrivenProblem(x, f(x))

b = Basis([polynomial_basis(z,2); exp10.(z)], z)

surrogates = _solve(p, f, 2, b, STLSQ(1e-1))

traverse(surrogates[3])

simplify(surrogates[4](z))