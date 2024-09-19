abstract type AbstractPathState end

struct PathState{T, PO <: Tuple, PI <: Tuple} <: AbstractPathState
    "Accumulated loglikelihood of the state"
    path_interval::Interval{T}
    "All the operators of the path"
    path_operators::PO
    "The unique identifier of nodes in the path"
    path_ids::PI

    function PathState{T}(
            interval::Interval{T}, path_operators::PO, path_ids::PI) where {T, PO, PI}
        return new{T, PO, PI}(interval, path_operators, path_ids)
    end
    function PathState{T}(
            interval::Interval, path_operators::PO, path_ids::PI) where {T, PO, PI}
        return new{T, PO, PI}(Interval{T}(interval), path_operators, path_ids)
    end
end

function PathState(interval::Interval{T}, id::Tuple{Int, Int} = (1, 1)) where {T}
    return PathState{T}(interval, (), (id,))
end

get_interval(state::PathState) = state.path_interval
get_operators(state::PathState) = state.path_operators
get_nodes(state::PathState) = state.path_ids

# Credit to https://discourse.julialang.org/t/efficient-tuple-concatenation/5398
@inline tuplejoin(x) = x
@inline tuplejoin(x, y) = (x..., y...)
@inline tuplejoin(x, y, z...) = tuplejoin(tuplejoin(x, y), z...)

function update_path(
        f::F where {F <: Function}, id::Tuple{Int, Int}, state::PathState{T}) where {T}
    return PathState{T}(
        f(get_interval(state)), (f, get_operators(state)...), (id, get_nodes(state)...))
end

function update_path(::Nothing, id::Tuple{Int, Int}, state::PathState{T}) where {T}
    return PathState{T}(
        get_interval(state), (identity, get_operators(state)...), (id, get_nodes(state)...))
end

function update_path(
        f::F where {F <: Function}, id::Tuple{Int, Int}, states::PathState{T}...) where {T}
    return PathState{T}(
        f(get_interval.(states)...), (f, tuplejoin(map(get_operators, states)...)...),
        (id, tuplejoin(map(get_nodes, states)...)...))
end

# Compute the degrees of freedom
get_dof(states::Vector{T}) where {T <: AbstractPathState} = length(get_nodes(states))

function get_nodes(states::Vector{T}) where {T <: AbstractPathState}
    return unique(reduce(vcat, map(collect ∘ get_nodes, states)))
end

check_intervals(p::AbstractPathState) = IntervalArithmetic.iscommon(get_interval(p))
