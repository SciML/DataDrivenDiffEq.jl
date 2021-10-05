using IntervalSets
using AbstractDifferentiation
using DataDrivenDiffEq
using ForwardDiff
using Statistics
using LinearAlgebra
## Data Intervals

ℜ_PD = Interval{:open, :open}(0.0, Inf)
ℜ_PSD = Interval{:closed, :open}(0.0, Inf)
ℜ = Interval{:open, :open}(-Inf, Inf)
ℜ_NSD = Interval{:open, :closed}(-Inf, 0.0)
ℜ_ND = Interval{:open, :open}(-Inf, 0.0)

function Base.in(x::AbstractArray, i::Interval)
    # Slight abuse for empty set
    isempty(x) && return false
    minimum(x) ∈ i && maximum(x) ∈ i
end

function Base.in(p::DataDrivenProblem, i::Interval)
    for f in [:DX, :X, :Y, :U]
        isempty(getfield(p, f)) && continue
        !(getfield(p, f) ∈ i) && return false
    end
    return true
end

function Base.in(p::DataDrivenProblem, i::Interval, s::Symbol)
    !(getfield(p, s) ∈ i) && return false
    return true
end

## Implement ForwardMode AD based on AbstractDifferentiation
# copied from https://github.com/JuliaDiff/AbstractDifferentiation.jl/blob/master/test/runtests.jl
struct ForwardDiffBackend <: AD.AbstractForwardMode end

AD.@primitive function jacobian(ab::ForwardDiffBackend, f, xs)
    if xs isa Number
        return (ForwardDiff.derivative(f, xs),)
    elseif xs isa AbstractArray
        out = f(xs)
        if out isa Number
            return (adjoint(ForwardDiff.gradient(f, xs)),)
        else
            return (ForwardDiff.jacobian(f, xs),)
        end
    elseif xs isa Tuple
        error(typeof(xs))      
    else
        error(typeof(xs)) 
    end
end

AD.primal_value(::ForwardDiffBackend, ::Any, f, xs) = ForwardDiff.value.(f(xs...))


## Surrogate analysis

mutable struct DataDrivenSurrogate{F, J}
    """Surrogate function"""
    f::F
    """Surrogate jacobian"""
    j::J
end

function DataDrivenSurrogate(f::Function, ad::AD.AbstractBackend = ForwardDiffBackend())
    j(x) = first(AD.jacobian(ad, f, x))
    return DataDrivenSurrogate(f, j)
end

(x::DataDrivenSurrogate)(xs...) = x.f(xs...)
∂(x::DataDrivenSurrogate, xs::AbstractVector) = x.j(xs)



f(x::AbstractVector) = [x[1]; x[2]+x[3]; exp(x[3]-x[1]); x[3]^2; prod(x)]
f(x::AbstractMatrix) = hcat(map(xi->f(xi), eachcol(x))...)

x = randn(3, 100)
y = f(x)

g(x::AbstractVector) = f(x)[3]
g(x::AbstractMatrix) = f(x)[3,:]

surrogate = DataDrivenSurrogate(f)
surrogate(x)
∂(surrogate, x[:,1])



function check_dependency(s::DataDrivenSurrogate, x::AbstractVector, tol = eps())
    j = ∂(s, x)
    normalize!.(eachrow(j), 1) # We normalize to sum_k j_ik = 1
    abs.(j) .> tol
end

function check_dependency(s::DataDrivenSurrogate, x::AbstractMatrix, tol = eps())
    round.(mean(map(xi->check_dependency(s, xi, tol), eachcol(x))))
end


# Check linearity using homogeneity
function check_linearity(s::DataDrivenSurrogate, x::AbstractMatrix{T}, alpha = convert(T, 1.2), tol = eps()) where T <: Number
    y = alpha*s(x) 
    x̂ = similar(x)
    x̂ .= x
    x̂ .*= alpha
    
    ŷ = s(x̂)

    idxs = zeros(Bool, size(y, 1))
    for i in 1:size(y, 1)
        idxs[i] = norm(y[i,:] .- ŷ[i,:]) / norm(y[i,:]) < tol
    end
    idxs
end


check_linearity(surrogate, x)



function check_invariances(s::DataDrivenSurrogate, x::AbstractMatrix{T}, 
    transform, inc, 
    alpha = one(T), tol = eps()) where T <: Number
    
    t, ops = transform

    y = s(x)
    ŷ = similar(y)

    x̂ = similar(x)

    _t = Vector{AbstractCoordinateTransform}(undef, size(y,1))
    for j in 1:size(x,1)-1, k in (j+1):size(x,1)
        x̂ .= x
        x̂[j, :] .= broadcast(ops[1], x[j, :], alpha)
        x̂[k, :] .= broadcast(ops[2], x[k, :], alpha)
        ŷ .= s(x̂)
        for i in 1:size(y,1)
            (inc[i,j] < 1 && inc[i,k] < 1) && continue
            
            if norm(ŷ[i,:] .- y[i,:]) / norm(y[i,:]) < tol
               _t[i] = CoordinateTransform(t, j, k)
            else
                _t[i] = IdentityTransform()
            end 
        end
    end
    _t
end

abstract type AbstractCoordinateTransform end

struct IdentityTransform <: AbstractCoordinateTransform end

(i::IdentityTransform)(x::AbstractVector) = x


struct CoordinateTransform <: AbstractCoordinateTransform
    """Transformation"""
    op::Function
    """First index"""
    i::Int
    """Second index"""
    j::Int
end

function (t::CoordinateTransform)(x::AbstractVector{T}) where T 
    x̂ = similar(x)
    for i in eachindex(x)
        if i == t.i
            x̂[i] = t.op(x[t.i], x[t.j])
        else
            x̂[i] = x[i]
        end
    end
    deleteat!(x̂, t.j)
    return x̂
end

(t::AbstractCoordinateTransform)(x::AbstractMatrix) = hcat(map(t, eachcol(x))...)

mutable struct InvestigationResult{T}
    """Dependency"""
    deps::AbstractMatrix{T}
    """Linear relations"""
    linearities::AbstractVector{Bool}
    """Input transformations"""
    transforms::AbstractArray{AbstractCoordinateTransform}
end

dependencies(r::InvestigationResult, idx, tol = eps()) = r.deps[idx, :] .> tol
is_linear(r::InvestigationResult, idx) = r.linearities[idx] > 0

function investigate_surrogate(s::DataDrivenSurrogate, x::AbstractMatrix{T}, 
    ops = Dict([
        (-, (+, +)),
        (+, (-, +)),
        (*, (*, /)),
        (/, (*, *))
    ]),
    alpha = convert(T, 1.2), tol = eps()) where T <: Number

    # Check dependencies
    deps = check_dependency(s, x, tol)
    # Check linearity 
    lins = check_linearity(s, x, alpha, tol)
    # Check invariants
    transforms = []
    for (a, b) in ops
        push!(transforms, check_invariances(s, x, (a,b), deps, alpha, tol))
    end

    transforms = hcat(transforms...)

    return InvestigationResult(deps, lins, transforms)
end

report = investigate_surrogate(surrogate, x)

dependencies(report, 1)

function apply_report(s::DataDrivenSurrogate, r::InvestigationResult, x::AbstractMatrix, tol = eps())
    y = s(x)
    idxs = dependencies(r, 1, tol)
    for i in 1:size(y, 1)
        idxs = dependencies(r, i, tol)
        if is_linear(r, i) 
            @show i y[i:i,:] / x[idxs, :]
        else

        end
    end
    # Remove all non-dependent variables for the corresponding y
    # Check if it is linear, if so apply linear regression (sparse regression)
    # If not, apply 
end

apply_report(surrogate, report, x)

report.linearities

prob = DiscreteDataDrivenProblem(randn(3,100), t = 0.0:1.0:99.0, U = rand(1, 100))


