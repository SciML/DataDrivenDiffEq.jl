using InverseFunctions: square 

function _safe_div(x::X, y::Y) where {X,Y} 
    iszero(y) && return zero(Y)
    \(x, y) 
end

InverseFunctions.inverse(::typeof(_safe_div)) = InverseFunctions.inverse(/)

function _safe_pow(x::X, y::Y) where {X,Y}
    iszero(x) ? x : ^(x, y)
end


InverseFunctions.inverse(::typeof(_safe_pow)) = InverseFunctions.inverse(^)

# We wrap a bunch of functions to complex, given that this way a NAN is returned
for f in (:sin, :cos, :log, :exp,  :sqrt, :square)
    sname = gensym(string(f))
    @eval begin
        ($sname)(x) = real($(f)(Complex(x)))
        ($sname)(x::Num) = $(f)(x)
        convert_to_safe(::typeof($f)) = $sname
        InverseFunctions.inverse(::typeof($sname)) = InverseFunctions.inverse($f)
    end
end

convert_to_safe(x) = x
convert_to_safe(::typeof(/)) = _safe_div
convert_to_safe(::typeof(^)) = _safe_pow
