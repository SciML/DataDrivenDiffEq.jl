struct DataDrivenFunction{IMPL, CTRLS, F1, F2} <: AbstractDataDrivenFunction{IMPL, CTRLS}
    f_oop::F1
    f_iip::F2
end

function DataDrivenFunction(rhs, implicits, states, parameters, iv, controls,
        eval_expression = false)
    _is_implicit = !isempty(implicits)
    _is_controlled = !isempty(controls)

    if !eval_expression
        f_oop, f_iip = build_function(rhs,
            value.(implicits), value.(states), value.(parameters),
            [value(iv)], value.(controls),
            expression = Val{false})
    else
        ex_oop, ex_iip = build_function(rhs,
            value.(implicits), value.(states),
            value.(parameters),
            [value(iv)], value.(controls),
            expression = Val{true})
        f_oop = eval(ex_oop)
        f_iip = eval(ex_iip)
    end

    return DataDrivenFunction{_is_implicit, _is_controlled, typeof(f_oop), typeof(f_iip)}(
        f_oop,
        f_iip)
end

_apply_function(f::DataDrivenFunction, du, u, p, t, c) = begin
    (; f_oop) = f
    f_oop(du, u, p, t, c)
end

function _apply_function!(f::DataDrivenFunction, res, du, u, p, t, c)
    (; f_iip) = f
    f_iip(res, du, u, p, t, c)
end

# Dispatch

# OOP 

# Without controls or implicits
function (f::DataDrivenFunction{false, false})(u::AbstractVector, p::P,
        t::Number) where {
        P <:
        Union{AbstractArray, Tuple
}}
    _apply_function(f, __EMPTY_VECTOR, u, p, t, __EMPTY_VECTOR)
end

# Without implicits, with controls
function (f::DataDrivenFunction{false, true})(u::AbstractVector, p::P, t::Number,
        c::AbstractVector) where {
        P <:
        Union{AbstractArray,
        Tuple}}
    _apply_function(f, __EMPTY_VECTOR, u, p, t, c)
end

# With implict, without controls
function (f::DataDrivenFunction{true, false})(du::AbstractVector, u::AbstractVector, p::P,
        t::Number) where {
        P <:
        Union{AbstractArray, Tuple}}
    _apply_function(f, du, u, p, t, __EMPTY_VECTOR)
end

# With implicit and controls
function (f::DataDrivenFunction{true, true})(du::AbstractVector, u::AbstractVector, p::P,
        t::Number,
        c::AbstractVector) where {
        P <:
        Union{AbstractArray,
        Tuple}}
    _apply_function(f, du, u, p, t, c)
end

# IIP 

# Without controls or implicits
function (f::DataDrivenFunction{false, false})(
        res::AbstractVector, u::AbstractVector, p::P,
        t::Number) where {
        P <:
        Union{AbstractArray, Tuple
}}
    _apply_function!(f, res, __EMPTY_VECTOR, u, p, t, __EMPTY_VECTOR)
end

# Without implicits, with controls
function (f::DataDrivenFunction{false, true})(res::AbstractVector, u::AbstractVector, p::P,
        t::Number,
        c::AbstractVector) where {
        P <:
        Union{AbstractArray,
        Tuple}}
    _apply_function!(f, res, __EMPTY_VECTOR, u, p, t, c)
end

# With implict, without controls
function (f::DataDrivenFunction{true, false})(res::AbstractVector, du::AbstractVector,
        u::AbstractVector, p::P,
        t::Number) where {
        P <:
        Union{AbstractArray, Tuple}}
    _apply_function!(f, res, du, u, p, t, __EMPTY_VECTOR)
end

# With implicit and controls
function (f::DataDrivenFunction{true, true})(res::AbstractVector, du::AbstractVector,
        u::AbstractVector, p::P, t::Number,
        c::AbstractVector) where {
        P <:
        Union{AbstractArray,
        Tuple}}
    _apply_function!(f, res, du, u, p, t, c)
end

##

## Matrix
maybeview(x::AbstractMatrix, id) = isempty(x) ? x : view(x, :, id)
maybeview(x::AbstractVector, id) = isempty(x) ? x : view(x, id)
maybeview(x, id) = x

function _check_array_inputs(res, du, u, p, t, c)
    # Collect the keys here
    n_obs = size(u, 2)
    @assert n_obs==length(t) "Number of observations $(n_obs) does not match timepoints $(length(t))"

    isempty(du) ||
        @assert n_obs==size(du, 2) "Number of observations $(n_obs) does not match implicits $(size(du, 2))"

    isempty(c) ||
        @assert n_obs==size(c, 2) "Number of observations $(n_obs) does not match controls $(size(c, 2))"

    isempty(res) ||
        @assert n_obs==size(res, 2) "Number of observations $(n_obs) does not match residuals $(size(res, 2))"
end

function _apply_vec_function(f::DataDrivenFunction, du::AbstractMatrix, u::AbstractMatrix,
        p::AbstractVector, t::AbstractVector, c::AbstractMatrix)
    _check_array_inputs(__EMPTY_MATRIX, du, u, p, t, c)

    reduce(hcat,
        map(axes(u, 2)) do i
            _apply_function(f,
                maybeview(du, i),
                maybeview(u, i),
                view(p, :),
                maybeview(t, i),
                maybeview(c, i))
        end)
end

function _apply_vec_function!(f::DataDrivenFunction, res::AbstractMatrix,
        du::AbstractMatrix, u::AbstractMatrix, p::AbstractVector,
        t::AbstractVector, c::AbstractMatrix)
    _check_array_inputs(res, du, u, p, t, c)

    foreach(axes(u, 2)) do i
        _apply_function!(f,
            maybeview(res, i),
            maybeview(du, i), maybeview(u, i), view(p, :),
            maybeview(t, i), maybeview(c, i))
    end
end

## OOP 

function (f::DataDrivenFunction{false, false})(u::AbstractMatrix, p::P,
        t::AbstractVector) where {
        P <: Union{
        AbstractArray,
        Tuple}}
    _apply_vec_function(f, __EMPTY_MATRIX, u, p, t, __EMPTY_MATRIX)
end

function (f::DataDrivenFunction{false, true})(u::AbstractMatrix, p::P, t::AbstractVector,
        c::AbstractMatrix) where {
        P <:
        Union{AbstractArray,
        Tuple}}
    _apply_vec_function(f, __EMPTY_MATRIX, u, p, t, c)
end

function (f::DataDrivenFunction{true, false})(du::AbstractMatrix, u::AbstractMatrix, p::P,
        t::AbstractVector) where {
        P <:
        Union{AbstractArray,
        Tuple}}
    _apply_vec_function(f, du, u, p, t, __EMPTY_MATRIX)
end

## IIP 

function (f::DataDrivenFunction{false, false})(
        res::AbstractMatrix, u::AbstractMatrix, p::P,
        t::AbstractVector) where {
        P <: Union{
        AbstractArray,
        Tuple}}
    _apply_vec_function!(f, res, __EMPTY_MATRIX, u, p, t, __EMPTY_MATRIX)
end

function (f::DataDrivenFunction{false, true})(res::AbstractMatrix, u::AbstractMatrix, p::P,
        t::AbstractVector,
        c::AbstractMatrix) where {
        P <:
        Union{AbstractArray,
        Tuple}}
    _apply_vec_function!(f, res, __EMPTY_MATRIX, u, p, t, c)
end

function (f::DataDrivenFunction{true, false})(res::AbstractMatrix, du::AbstractMatrix,
        u::AbstractMatrix, p::P,
        t::AbstractVector) where {
        P <:
        Union{AbstractArray,
        Tuple}}
    _apply_vec_function!(f, res, du, u, p, t, __EMPTY_MATRIX)
end
