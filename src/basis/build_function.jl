# Helper to build function
# TODO eval -> Runtime generated
function _build_ddd_function(rhs, states, parameters, iv, eval_expression::Bool = false)

    if eval_expression
        f_oop, f_iip = eval.(build_function(rhs, value.(states), value.(parameters), [value(iv)], expression = Val{true}))
    else
        f_oop, f_iip = build_function(rhs, value.(states), value.(parameters), [value(iv)], expression = Val{false})
    end

    function f(
        u::AbstractVector{T} where T,
        p::AbstractVector{T} where T,
        t::T where T
    )
        return f_oop(u, p, t)
    end

    function f(
        du::AbstractVector{T} where T,
        u::AbstractVector{T} where T,
        p::AbstractVector{T} where T,
        t::T where T
    )
        return f_iip(du, u, p, t)
    end

    function f(
        x::AbstractMatrix{T} where T,
        p::AbstractVector{T} where T,
        t::AbstractVector{T} where T
    )
        @assert size(x, 2) == length(t) "Measurements and time points must be of equal length!"

        return hcat([f(x[:,i], p, t[i]) for i in 1:size(x, 2)]...)

    end


    function f(
        y::AbstractMatrix{T} where T,
        x::AbstractMatrix{T} where T,
        p::AbstractVector{T} where T,
        t::AbstractVector{T} where T
    )
        @assert size(x, 2) == length(t) "Measurements and time points must be of equal length!"
        @assert size(x, 2) == size(y, 2) "Measurements and preallocated output must be of equal length!"

        for i = 1:size(x, 2)
            @views f(y[:, i], x[:, i], p, t[i])
        end

        return
    end


    # Dispatch on DiffEqBase.NullParameters
    f(u,p::DiffEqBase.NullParameters, t) = f(u,[], t)
    f(du, u,p::DiffEqBase.NullParameters, t) = f(du, u,[], t)

    # And on the controls
    f(u,p,t,input) = f(u,p,t)
    f(du,u,p,t,input) = f(du,u,p,t)

    return f
end


function _build_ddd_function(
    rhs,
    states,
    parameters,
    iv,
    controls,
    eval_expression::Bool = false,
)

    isempty(controls) &&
        return _build_ddd_function(rhs, states, parameters, iv, eval_expression)

    # Assumes zero control is zero!

    if eval_expression
        c_oop, c_iip =
            eval.(
                build_function(
                    rhs,
                    value.(states),
                    value.(parameters),
                    [value(iv)],
                    value.(controls),
                    expression = Val{true},
                ),
            )
    else
        c_oop, c_iip = build_function(
            rhs,
            value.(states),
            value.(parameters),
            [value(iv)],
            value.(controls),
            expression = Val{false},
        )
    end

    function f(
        u::AbstractVector{T} where T,
        p::AbstractVector{T} where T,
        t::T where T,
        c::AbstractVector{T} where T = zeros(eltype(u), size(controls)...),
    )
        return c_oop(u, p, t, c)
    end

    function f(
        du::AbstractVector{T} where T,
        u::AbstractVector{T} where T,
        p::AbstractVector{T} where T,
        t::T where T,
        c::AbstractVector{T} where T= zeros(eltype(u), size(controls)...),
    )
        return c_iip(du, u, p, t, c)
    end


    function f(
        x::AbstractMatrix{T} where T,
        p::AbstractVector{T} where T,
        t::AbstractVector{T} where T
    )
        @assert size(x, 2) == length(t) "Measurements and time points must be of equal length!"

        return hcat([f(x[:,i], p, t[i]) for i in 1:size(x, 2)]...)

    end

    function f(
        y::AbstractMatrix{T} where T,
        x::AbstractMatrix{T} where T,
        p::AbstractVector{T} where T,
        t::AbstractVector{T} where T
    )
        @assert size(x, 2) == length(t) "Measurements and time points must be of equal length!"
        @assert size(x, 2) == size(y, 2) "Measurements and preallocated output must be of equal length!"

        for i = 1:size(x, 2)
            @views f(y[:, i], x[:, i], p, t[i])
        end

        return
    end


    function f(
        x::AbstractMatrix{T} where T,
        p::AbstractVector{T} where T,
        t::AbstractVector{T} where T,
        u::AbstractMatrix{T} where T
    )
        @assert size(x, 2) == length(t) "Measurements and time points must be of equal length!"
        @assert size(x, 2) == size(u, 2) "Measurements and inputs must be of equal length!"


        return hcat([f(x[:,i], p, t[i], u[:, i]) for i in 1:size(x, 2)]...)

    end

    function f(
        y::AbstractMatrix{T} where T,
        x::AbstractMatrix{T} where T,
        p::AbstractVector{T} where T,
        t::AbstractVector{T} where T,
        u::AbstractMatrix{T} where T
    )
        @assert size(x, 2) == length(t) "Measurements and time points must be of equal length!"
        @assert size(x, 2) == size(y, 2) "Measurements and preallocated output must be of equal length!"
        @assert size(x, 2) == size(u, 2) "Measurements and inputs must be of equal length!"

        for i = 1:size(x, 2)
            @views f(y[:, i], x[:, i], p, t[i], u[:, i])
        end

        return
    end

    # Dispatch on DiffEqBase.NullParameters
    f(u,p::DiffEqBase.NullParameters, t) = f(u,[], t)
    f(du, u,p::DiffEqBase.NullParameters, t) = f(du, u,[], t)


    return f
end
