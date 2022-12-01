_safe_div(x, y::T) where {T} = begin
    (iszero(y) && iszero(x)) && return one(T) # one(T)
    iszero(y) && return zero(T)
    /(x, y)
end

_safe_sqrt(x::T) where {T} = real(sqrt(Complex(x)))
_safe_sqrt(x::Num) = sqrt(x)

_safe_log(x::T) where {T <: Number} = real(log(Complex(x)))
_safe_log(x::Num) = log(x)

_safe_pow(x::T, y) where {T} = begin
    iszero(x) && return zero(T)
    ^(x, y)
end

"""
$(SIGNATURES)

Convert a given function to its protected counterpart.

Currently, the following functions are proteced:
+ `/` 
+ `sqrt`
+ `log`
+ `^`
"""
convert_to_safe(x) = x
convert_to_safe(::typeof(/)) = _safe_div
convert_to_safe(::typeof(log)) = _safe_log
convert_to_safe(::typeof(sqrt)) = _safe_sqrt
convert_to_safe(::typeof(^)) = _safe_pow
