mutable struct ParetoCandidate{U, V, W}
    point::AbstractArray{U}
    parameter::AbstractArray{U}
    iter::V
    threshold::W
end

point(x::ParetoCandidate) = x.point
parameter(x::ParetoCandidate) = x.parameter
iter(x::ParetoCandidate) = x.iter
threshold(x::ParetoCandidate) = x.threshold

function update!(p::ParetoCandidate, point, parameter, iter, threshold)
    p.point .= point
    p.parameter .= parameter
    p.iter = iter
    p.threshold = threshold
    return
end

function update!(p::ParetoCandidate, y::ParetoCandidate)
    p.point .= point(y)
    p.parameter .= parameter(y)
    p.iter = iter(y)
    p.threshold = threshold(y)
    return
end


mutable struct WeightedSum <: AbstractScalarizationMethod
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

struct GoalProgramming <: AbstractScalarizationMethod
    f::Function
    n::Function
end

GoalProgramming() = GoalProgramming(x->norm(x, 2), x->identity(x))

(g::GoalProgramming)(x::ParetoCandidate) = g.n(g.f(point(x)))

mutable struct WeightedExponentialSum <: AbstractScalarizationMethod
    w::Union{UniformScaling, AbstractArray}
    f::Function
    p::Real
end

WeightedExponentialSum() = WeightedExponentialSum(I, identity, 2)

weights(x::WeightedExponentialSum) = x.w

function weights!(x::WeightedExponentialSum, w::AbstractArray)
    x.w = w
    return
end

(w::WeightedExponentialSum)(x::ParetoCandidate) = sum(w.w*w.f(point(x)).^w.p)


mutable struct ParetoFront{S}
    candidates::AbstractArray{ParetoCandidate}
    scalarization::S
end

candidates(x::ParetoFront) = x.candidates

function ParetoFront(n::Int64; scalarization::AbstractScalarizationMethod = WeightedSum())
    candidates = Array{ParetoCandidate}(undef, n)
    return ParetoFront(candidates, scalarization)
end

function Base.empty!(x::ParetoFront)
    x.candidates .= Array{ParetoCandidate}(undef, size(candidates(x))...)
    return
end

(x::ParetoFront)(y::ParetoCandidate) = x.scalarization(y)

function assert_dominance(x::ParetoFront, y::ParetoFront)
    [x(cx) < y(cy) for (cx, cy) in zip(candidates(x), candidates(y))]
end

function conditional_add!(x::ParetoFront, y::ParetoFront)
    for (i, c) in enumerate(assert_dominance(y, x))
        c ? update!(x.candidates[i], y.candidates[i]) : nothing
    end
end

function set_candidate!(x::ParetoFront, idx, point, parameter, iter, threshold)
    if isdefined(candidates(x), idx)
        update!(x.candidates[idx], point, parameter, iter, threshold)
    else
        x.candidates[idx] = ParetoCandidate(point, parameter, iter, threshold)
    end
    return
end

Base.getindex(x::ParetoFront, idx) = getindex(x.candidates, idx)
iter(x::ParetoFront) =  maximum(iter.(candidates(x)))
threshold(x::ParetoFront) = minimum(threshold.(candidates(x)))
