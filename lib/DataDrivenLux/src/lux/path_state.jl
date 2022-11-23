abstract type AbstractPathState end

struct PathState{T} <: AbstractPathState
    "Accumulated loglikelihood of the state"
    path_loglikelihood::T
    "All the operators of the path"
    path_operators::Tuple
    "The unique identifier of nodes in the path"
    path_ids::Tuple
end

function PathState(initial_ll::T, id::Int = 0) where T
    return PathState{T}(initial_ll, (), (id,))
end

get_loglikelihood(state::PathState) = state.path_loglikelihood
get_operators(state::PathState) = state.path_operators
get_nodes(state::PathState) = state.path_ids

# Credit to https://discourse.julialang.org/t/efficient-tuple-concatenation/5398
@inline tuplejoin(x) = x
@inline tuplejoin(x, y) = (x..., y...)
@inline tuplejoin(x, y, z...) = tuplejoin(tuplejoin(x, y), z...)

function update_path(f::Function, ll::T, id::Tuple{Int,Int}, state::PathState{T}) where T
    PathState{T}(
        ll + get_loglikelihood(state),
        (f, get_operators(state)...), 
        (id, get_nodes(state)...)
    )
end

function update_path(::Nothing, ll::T,  id::Tuple{Int,Int}, state::PathState{T}) where T
    PathState{T}(
        ll + get_loglikelihood(state),
        (identity, get_operators(state)...),
        (id, get_nodes(state)...)
    )
end

function update_path(f::Function, ll::T, id::Tuple{Int,Int}, states::PathState{T}...) where T
    PathState{T}(
        ll + sum(get_loglikelihood, states),
        (f, tuplejoin(map(get_operators, states)...)...),
        (id, tuplejoin(map(get_nodes, states)...)...)
    )
end

# Compute the degrees of freedom
get_dof(states::Vector{T}) where T <: AbstractPathState = length(get_nodes(states))
get_nodes(states::Vector{T}) where T <: AbstractPathState = unique(reduce(vcat, map(collect ∘ get_nodes, states)))