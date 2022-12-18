using InverseFunctions: square

function _safe_div(x::X, y::Y) where {X, Y}
    iszero(y) && return zero(Y)
    \(x, y)
end

InverseFunctions.inverse(::typeof(_safe_div)) = _safe_div

function _safe_pow(x::X, y::Y) where {X, Y}
    iszero(x) ? x : ^(x, y)
end

InverseFunctions.inverse(::typeof(_safe_pow)) = InverseFunctions.inverse(^)

safe_functions = Dict(f => gensym(string(f)) for f in (sin, cos, log, exp, sqrt, square))

inverse_safe = Dict()

for (f, safe_f) in safe_functions
    finv = InverseFunctions.inverse(f)
    @info f finv
    if isa(finv, InverseFunctions.NoInverse)
        inverse_safe[safe_f] = NoInverse(safe_f)
    else
        inverse_safe[safe_f] = safe_functions[finv]
    end
end

for (f, sname) in safe_functions
    @eval begin
        ($sname)(x) = real($(f)(Complex(x)))
        ($sname)(x::Num) = $(f)(x)
        convert_to_safe(::typeof($f)) = $sname
        InverseFunctions.inverse(::typeof($sname)) = $(inverse_safe[sname])
    end
end
convert_to_safe(x) = x
convert_to_safe(::typeof(/)) = _safe_div
convert_to_safe(::typeof(^)) = _safe_pow
