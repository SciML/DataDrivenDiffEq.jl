struct ParetoCandidate
    point
    parameter
end

point(x::ParetoCandidate) = x.point
parameter(x::ParetoCandidate) = x.parameter

abstract type AbstractSortingMethod end;

mutable struct ParetoFront{S}
    candidates::AbstractArray{ParetoCandidate}
    sorting::S
end


function apply!(x::ParetoFront, y::AbstractSortingMethod)
    p = sortperm(candidates(x), by = x->y(x))
    x.candidates .= candidates(x)[p]
    return p[1]
end

function Base.empty!(x::ParetoFront)
    x.candidates = ParetoCandidate[]
    return
end

function add_candidate!(x::ParetoFront, point, parameter)
    push!(x.candidates, ParetoCandidate(point, paremeter))
    return
end

function add_candidate!(x::ParetoFront, y::ParetoCandidate)
    push!(x.candidates, y)
end

candidates(x::ParetoFront) = x.candidates
sorting(x::ParetoFront) = x.sorting
best(x::ParetoFront) = x.candidates[1]

function Base.sort!(x::ParetoFront)
    apply!(x, x.sorting)
end

struct Domination <: AbstractSortingMethod
    f::Function
end

Domination() = Domination((x)->norm(x, 2))

(d::Domination)(x::ParetoCandidate) = d.f(point(x))


ParetoFront(; sorting = Domination()) = ParetoFront(ParetoCandidate[], sorting)

mutable struct WeightedSum <: AbstractSortingMethod
    w::Union{UniformScaling, AbstractArray}
    f::Function
end

WeightedSum() = WeightedSum(I, x->identity(x))

weights(x::WeightedSum) = x.w

function weights!(x::WeightedSum, w::AbstractArray)
    x.w = w
    return
end

(w::WeightedSum)(x::ParetoCandidate) = sum(w.w*w.f(point(x)))

struct GoalProgramming <: AbstractSortingMethod
    f::Function
    n::Function
end

GoalProgramming() = GoalProgramming(x->norm(x, 2), x->identity(x))

(g::GoalProgramming)(x::ParetoCandidate) = g.n(g.f(point(x)))

mutable struct WeigthedExponentialSum <: AbstractSortingMethod
    w::Union{UniformScaling, AbstractArray}
    f::Function
    p::Real
end

WeigthedExponentialSum() = WeigthedExponentialSum(I, identity, 2)

weights(x::WeigthedExponentialSum) = x.w

function weights!(x::WeigthedExponentialSum, w::AbstractArray)
    x.w = w
    return
end

(w::WeigthedExponentialSum)(x::ParetoCandidate) = sum(w.w*w.f(point(x)).^w.p)
