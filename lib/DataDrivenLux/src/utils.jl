function _safe_div(x::X, y::Y) where {X,Y} 
    (iszero(x) && iszero(y)) && return one(X)
    iszero(y) && return zero(Y)
    \(x, y) 
end


function _safe_pow(x::X, y::Y) where {X,Y}
    iszero(x) ? x : ^(x, y)
end

# We wrap a bunch of functions to complex, given that this way a NAN is returned
for f in (:sin, :cos, :log, :exp,  :sqrt)
    sname = gensym(string(f))
    @eval begin
        ($sname)(x) = real($(f)(Complex(x)))
        ($sname)(x::Num) = $(f)(x)
        convert_to_safe(::typeof($f)) = $sname
    end
end

#_safe_sin(x::X) where X = isinf(x) ? NaN : sin(x)
#_safe_cos(x::X) where X = isinf(x) ? NaN : cos(x)
#_safe_log(x::X) where X = x <= zero(X) ? real(log(Complex(x))) : log(x)
#_safe_sqrt(x::X) where X = x <= zero(X) ? NaN : sqrt(x)
#_safe_pow(x::X, y::Y) where {X,Y} = iszero(x) ? NaN : ^(x, y)


convert_to_safe(x) = x
convert_to_safe(::typeof(/)) = _safe_div
convert_to_safe(::typeof(^)) = _safe_pow
