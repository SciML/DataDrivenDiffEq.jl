struct ParetoCandidate
    objective
    point
end

objective(x::ParetoCandidate) = x.objective
point(x::ParetoCandidate) = x.point

function dominates(x::ParetoCandidate, y::AbstractArray{ParetoCandidate})
    all(objective(x) .<= objective.(y)) #&& any(objective(x) .< objective.(y))
end



abstract type AbstractSortingMethod end;

mutable struct ParetoFront{S}
    candidates::AbstractArray{ParetoCandidate}
    objective::Function
    sorting::S
    best::AbstractArray{Int64}
end


ParetoFront(; objective = x->norm(x, 2)) = ParetoFront(ParetoCandidate[], objective, Domination(), Int64[])

function ParetoFront(X::AbstractArray; objective = (x)->norm(x,2))
    candidates = ParetoCandidate[]
    for xi in eachcol(X)
        push!(candidates, ParetoCandidate(objective(xi), xi))
    end
    return ParetoFront(candidates, objective, Domination(), Int64[])
end


candidates(x::ParetoFront) = x.candidates
objective(x::ParetoFront) = x.objective
sorting(x::ParetoFront) = x.sorting
best(x::ParetoFront) = x.best
scores(x::ParetoFront) = objective.(candidates(x))

function Base.sort!(x::ParetoFront)
    apply!(x, x.sorting)
end


struct Domination <: AbstractSortingMethod end;

function apply!(x::ParetoFront, y::Domination)
    best = Int64[]
    c = candidates(x)
    for (i, ci) in enumerate(c)
        d = dominates(ci, c)
        if d
            push!(best, i)
        end
    end
    x.best = best
    return
end

struct WeightedSum <: AbstractSortingMethod
    w::AbstractArray
    f::Function
end

(w::WeightedSum)(x) = sum(w.*f(x))

function apply!(x::ParetoFront, y::WeightedSum)
    fs = zeros(eltype(y.w), length(x.candidates))
    for (i, c) in enumerate(candidates(x))
        fs[i] = y(point(c))
    end

    _, idxs = findmin(fs)
    x.best = [idxs]
    return
end

f = 
