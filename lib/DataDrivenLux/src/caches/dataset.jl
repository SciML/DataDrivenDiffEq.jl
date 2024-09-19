struct Dataset{T}
    x::AbstractMatrix{T}
    y::AbstractMatrix{T}
    u::AbstractMatrix{T}
    t::AbstractVector{T}
    x_intervals::AbstractVector{Interval{T}}
    y_intervals::AbstractVector{Interval{T}}
    u_intervals::AbstractVector{Interval{T}}
    t_interval::Interval{T}
end

Base.eltype(::Dataset{T}) where {T} = T

function Dataset(X::AbstractMatrix, Y::AbstractMatrix,
        U::AbstractMatrix = Array{eltype(X)}(undef, 0, 0),
        t::AbstractVector = Array{eltype(X)}(undef, 0))
    T = Base.promote_eltype(X, Y, U, t)
    X = convert.(T, X)
    Y = convert.(T, Y)
    U = convert.(T, U)
    t = convert.(T, t)
    t = isempty(t) ? convert.(T, LinRange(0, size(Y, 2) - 1, size(Y, 2))) : convert.(T, t)
    x_intervals = Interval.(map(extrema, eachrow(X)))
    y_intervals = Interval.(map(extrema, eachrow(Y)))
    u_intervals = Interval.(map(extrema, eachrow(U)))
    t_intervals = isempty(t) ? Interval{T}(zero(T), zero(T)) : Interval(extrema(t))
    return Dataset{T}(X, Y, U, t, x_intervals, y_intervals, u_intervals, t_intervals)
end

function Dataset(prob::DataDrivenDiffEq.DataDrivenProblem)
    X, _, t, U = DataDrivenDiffEq.get_oop_args(prob)
    Y = DataDrivenDiffEq.get_implicit_data(prob)
    return Dataset(X, Y, U, t)
end

function (b::Basis{false, false})(d::Dataset{T}, p::P) where {T, P}
    f = DataDrivenDiffEq.get_f(b)
    (; x, t) = d
    return f(x, p, t)
end

function (b::Basis{false, true})(d::Dataset{T}, p::P) where {T, P}
    f = DataDrivenDiffEq.get_f(b)
    (; x, t, u) = d
    return f(x, p, t, u)
end

function (b::Basis{true, false})(d::Dataset{T}, p::P) where {T, P}
    f = DataDrivenDiffEq.get_f(b)
    (; y, x, t) = d
    return f(y, x, p, t)
end

function (b::Basis{true, true})(d::Dataset{T}, p::P) where {T, P}
    f = DataDrivenDiffEq.get_f(b)
    (; y, x, t, u) = d
    return f(y, x, p, t, u)
end

##

function interval_eval(b::Basis{false, false}, d::Dataset{T}, p::P) where {T, P}
    f = DataDrivenDiffEq.get_f(b)
    (; x_intervals, t_interval) = d
    return f(x_intervals, p, t_interval)
end

function interval_eval(b::Basis{false, true}, d::Dataset{T}, p::P) where {T, P}
    f = DataDrivenDiffEq.get_f(b)
    (; x_intervals, t_interval, u_intervals) = d
    return f(x_intervals, p, t_interval, u_intervals)
end

function interval_eval(b::Basis{true, false}, d::Dataset{T}, p::P) where {T, P}
    f = DataDrivenDiffEq.get_f(b)
    (; y_intervals, x_intervals, t_interval) = d
    return f(y_intervals, x_intervals, p, t_interval)
end

function interval_eval(b::Basis{true, true}, d::Dataset{T}, p::P) where {T, P}
    f = DataDrivenDiffEq.get_f(b)
    (; y_intervals, x_intervals, t_interval, u_intervals) = d
    return f(y_intervals, x_intervals, p, t_interval, u_intervals)
end
