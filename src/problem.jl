using DiffEqBase
using DataDrivenDiffEq
using DataInterpolations
using LinearAlgebra

include(joinpath(pwd(), "src" , "utils", "collocation.jl"))

function _promote(args...)
    _type = Base.promote_eltype(args...)
    return map(x->convert.(_type, x), args)
end

_isfun(x) = false
_isfun(x::F) where F <: Function = true

abstract type AbstractDataDrivenProblem end

"""
$(TYPEDEF)

The `DataDrivenProblem` defines

# Fields
$(FIELDS)

# Example

```julia
X, DX, t = data...
prob = DiscreteDataDrivenProblem(X)
prob = ContinuousDataDrivenProblem(X, DX)
prob = ContinuousDataDrivenProblem(X, t)

input_signal(u,p,t) = t^2
prob = DiscreteDataDrivenProblem(X, )
```
"""
struct DataDrivenProblem{dType, P, uType} <: AbstractDataDrivenProblem where {uType <: Union{AbstractMatrix, Function}}

    # Data
    """State measurements"""
    X::AbstractMatrix{dType}
    """Time measurements (optional)"""
    t::AbstractVector{dType}
    """Differental state measurements (optional)"""
    DX::AbstractMatrix{dType}
    """Output measurements (optional; not used right now) """
    Y::AbstractMatrix{dType}
    """Input measurements (optional) provided either as an `AbstractArray` or a `Function` of form `f(u,p,t)`"""
    U::uType
    
    """(Time) discrete problem"""
    is_discrete::Bool

    function DataDrivenProblem(
        X::AbstractMatrix, t::AbstractVector, DX::AbstractMatrix,
        Y::AbstractMatrix, U::AbstractMatrix, is_discrete::Bool
        )
        dType = Base.promote_eltype(X, t, DX, Y, U)
        return new{dType, typeof(p), typeof(U)}(_promote(X, t, DX, Y, U)..., is_discrete)
    end

    function DataDrivenProblem(
        X::AbstractMatrix, t::AbstractVector, DX::AbstractMatrix,
        Y::AbstractMatrix, U::AbstractMatrix, p::AbstractArray, b::Basis, is_discrete::Bool
        )
        dType = Base.promote_eltype(X, t, DX, Y, U, p)
        return new{dType, typeof(p), typeof(U)}(_promote(X, t, DX, Y, U, p)..., b, is_discrete)
    end

    function DataDrivenProblem(
        X::AbstractMatrix, t::AbstractVector, DX::AbstractMatrix,
        Y::AbstractMatrix, U::F, p, b::Basis, is_discrete::Bool
        ) where F <: Function
        dType = Base.promote_eltype(X, t, DX, Y)
        return new{dType, typeof(p), typeof(U)}(_promote(X, t, DX, Y)..., U ,p, b, is_discrete)
    end

    function DataDrivenProblem(
        X::AbstractMatrix, t::AbstractVector, DX::AbstractMatrix,
        Y::AbstractMatrix, U::F, p::AbstractArray, b::Basis, is_discrete::Bool
        ) where F <: Function
        dType = Base.promote_eltype(X, t, DX, Y)
        p = convert.(dType, p)
        return new{dType, typeof(p), typeof(U)}(_promote(X, t, DX, Y)..., U , p, b, is_discrete)
    end
end

init_estimate() = x.basis(x.X, x.p, x.t)

DiscreteDataDrivenProblem(args...) = DataDrivenProblem(args..., true)
ContinuousDataDrivenProblem(args...) = DataDrivenProblem(args..., false)


## Discrete

function DiscreteDataDrivenProblem(X::AbstractMatrix, t::AbstractVector,
    U::uType) where uType <: Union{AbstractMatrix, Function}
    return DiscreteDataDrivenProblem(
        X, t, Array{eltype(X)}(undef, 0,0), Array{eltype(X)}(undef, 0,0), U, Basis(size(X, 1)), DiffEqBase.NullParamters()
    )
end

function DiscreteDataDrivenProblem(X::AbstractArray, b::Basis = Basis(size(X, 1)),p = DiffEqBase.NullParameters())
    return DiscreteDataDrivenProblem(X, Array{eltype(X)}(undef, 0), Array{eltype(X)}(undef, size(X, 1), 0), Array{eltype(X)}(undef, 0, 0), Array{eltype(X)}(undef, 0, 0), p, b)
end

function DiscreteDataDrivenProblem(X::AbstractArray, t::AbstractVector, b::Basis = Basis(size(X, 1)),p = DiffEqBase.NullParameters())
    return DiscreteDataDrivenProblem(X, t, Array{eltype(X)}(undef, size(X, 1), 0), Array{eltype(X)}(undef, 0, 0), Array{eltype(X)}(undef, 0, 0), p, b)
end

function DiscreteDataDrivenProblem(X::AbstractArray, t::AbstractVector, U::AbstractArray, b::Basis = Basis(size(X, 1)),p = DiffEqBase.NullParameters())
    return DiscreteDataDrivenProblem(X, t, Array{eltype(X)}(undef, size(X, 1), 0), Array{eltype(X)}(undef, 0, 0), U, p, b)
end

function DiscreteDataDrivenProblem(X::AbstractArray, t::AbstractVector, ::F, b::Basis = Basis(size(X, 1)),p = DiffEqBase.NullParameters()) where F <: Function
    return DiscreteDataDrivenProblem(X, t, Array{eltype(X)}(undef, size(X, 1), 0), Array{eltype(X)}(undef, 0, 0),U, p, b)
end

## Continuous

function ContinuousDataDrivenProblem(X::AbstractMatrix, t::AbstractVector, DX::AbstractMatrix,
    U, b::Basis = Basis(size(X, 1)), p = DiffEqBase.NullParamters())
    return ContinuousDataDrivenProblem(
        X, t, DX, Array{eltype(X)}(undef, 0,0), U, b, p
    )
end

function ContinuousDataDrivenProblem(X::AbstractMatrix, DX::AbstractMatrix, b::Basis = Basis(size(X, 1)), p = DiffEqBase.NullParameters())
    @assert size(X) == size(DX)
    return ContinuousDataDrivenProblem(X, Array{eltype(X)}(undef, 0), DX, Array{eltype(X)}(undef, 0, 0), Array{eltype(X)}(undef, 0, 0), p, b)
end

function ContinuousDataDrivenProblem_(X::AbstractMatrix, t::AbstractVector, b::Basis = Basis(size(X, 1)),p = DiffEqBase.NullParameters();
    collocation = TriangularKernel(), tsample::AbstractVector = t)
    dx_, x_ = collocate_data(X, t, collocation)
    return ContinuousDataDrivenProblem(x_, t, dx_, Array{eltype(X)}(undef, 0, 0), Array{eltype(X)}(undef, 0, 0), p, b)
end

function ContinuousDataDrivenProblem(X::AbstractMatrix, t::AbstractVector, U::AbstractMatrix, b::Basis = Basis(size(X, 1)),p = DiffEqBase.NullParameters();
    collocation = TriangularKernel(), tsample::AbstractVector = t)
    dx_, x_ = collocate_data(X, t, collocation)
    return ContinuousDataDrivenProblem(x_, t, dx_, Array{eltype(X)}(undef, 0, 0), Array{eltype(X)}(undef, 0, 0), p, b)
end

function ContinuousDataDrivenProblem(X::AbstractMatrix, t::AbstractVector, U::F, b::Basis = Basis(size(X, 1)),p = DiffEqBase.NullParameters();
    collocation = TriangularKernel(), tsample::AbstractVector = t) where F <: Function
    dx_, x_ = collocate_data(X, t, collocation)
    return ContinuousDataDrivenProblem(x_, t, dx_, Array{eltype(X)}(undef, 0, 0), U, p, b)
end


t = collect(0:0.1:6.0)
X = vcat(sin.(t'), cos.(t'))
DiscreteDataDrivenProblem(X, t, (u,p,t)->[sin(u[1])])
DX = vcat(cos.(t'), -sin.(t'))
prob = DiscreteDataDrivenProblem(X, t)
prob = ContinuousDataDrivenProblem(X, t)
prob()
all(isapprox.(prob.DX, DX, atol = 1e-1))
itp = InterpolationMethod(CubicSpline)
dx, x = collocate_data(X, t, t, itp)
prob = ContinuousDataDrivenProblem(X,t, collocation = itp)
typeof((u,p,t)->[sin(t)]) <: Function
prob = ContinuousDataDrivenProblem(X,t,(u,p,t)->[sin(t)], collocation = itp)
all(isapprox.(prob.DX, DX, atol = 1e-1))
