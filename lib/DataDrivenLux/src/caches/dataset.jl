@concrete struct Dataset{T}
    x <: AbstractMatrix{T}
    y <: AbstractMatrix{T}
    u <: AbstractMatrix{T}
    t <: AbstractVector{T}
    x_intervals <: AbstractVector{Interval{T}}
    y_intervals <: AbstractVector{Interval{T}}
    u_intervals <: AbstractVector{Interval{T}}
    t_interval <: Interval{T}
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
    x_intervals = interval.(map(extrema, eachrow(X)))
    y_intervals = interval.(map(extrema, eachrow(Y)))
    u_intervals = interval.(map(extrema, eachrow(U)))
    t_intervals = isempty(t) ? Interval{T}(zero(T), zero(T)) : interval(extrema(t))
    return Dataset{T}(X, Y, U, t, x_intervals, y_intervals, u_intervals, t_intervals)
end

function Dataset(prob::DataDrivenDiffEq.DataDrivenProblem)
    X, _, t, U = DataDrivenDiffEq.get_oop_args(prob)
    Y = DataDrivenDiffEq.get_implicit_data(prob)
    return Dataset(X, Y, U, t)
end

function (b::Basis{false, false})(d::Dataset{T}, p::P) where {T, P}
    f = DataDrivenDiffEq.get_f(b)
    return f(d.x, p, d.t)
end

function (b::Basis{false, true})(d::Dataset{T}, p::P) where {T, P}
    f = DataDrivenDiffEq.get_f(b)
    return f(d.x, p, d.t, d.u)
end

function (b::Basis{true, false})(d::Dataset{T}, p::P) where {T, P}
    f = DataDrivenDiffEq.get_f(b)
    return f(d.y, d.x, p, d.t)
end

function (b::Basis{true, true})(d::Dataset{T}, p::P) where {T, P}
    f = DataDrivenDiffEq.get_f(b)
    return f(d.y, d.x, p, d.t, d.u)
end

function interval_eval(b::Basis{false, false}, d::Dataset{T}, p::P) where {T, P}
    f = DataDrivenDiffEq.get_f(b)
    return f(d.x_intervals, p, d.t_interval)
end

function interval_eval(b::Basis{false, true}, d::Dataset{T}, p::P) where {T, P}
    f = DataDrivenDiffEq.get_f(b)
    return f(d.x_intervals, p, d.t_interval, d.u_intervals)
end

function interval_eval(b::Basis{true, false}, d::Dataset{T}, p::P) where {T, P}
    f = DataDrivenDiffEq.get_f(b)
    return f(d.y_intervals, d.x_intervals, p, d.t_interval)
end

function interval_eval(b::Basis{true, true}, d::Dataset{T}, p::P) where {T, P}
    f = DataDrivenDiffEq.get_f(b)
    return f(d.y_intervals, d.x_intervals, p, d.t_interval, d.u_intervals)
end
