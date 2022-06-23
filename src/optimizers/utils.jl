"""
$(SIGNATURES)

Clips the solution by the given threshold `λ` and ceils the entries to the corresponding decimal.
"""
function clip_by_threshold!(x::AbstractArray, λ::T, rounding::Bool = true) where {T<:Real}
    #dplace = ceil(Int, -log10(λ))
    for i in eachindex(x)
        x[i] = abs(x[i]) < λ ? zero(eltype(x)) : x[i]
        #x[i] = rounding ? round(x[i], digits = dplace) : x[i]
    end
    return
end


# Evaluate the results for pareto
G(opt::AbstractOptimizer{T} where {T}) = f -> f[1] < 1 ? Inf : norm(f, 2) # 2*f[1]-2*log(f[2])
G(opt::AbstractSubspaceOptimizer{T} where {T}) = f -> f[1] < 2 ? Inf : norm(f, 2) # 2*f[1]-2*log(f[2])
# Evaluate F
function F(opt::AbstractOptimizer{T} where {T})
    f(x, A, y::AbstractArray) = [norm(x, 0); norm(y .- A * x, 2)] # explicit
    f(x, A, y::AbstractArray, λ) = [norm(x, 0, λ); norm(y .- A * x, 2, λ)]
    f(x, A) = [norm(x, 0); norm(A * x, 2)] # implicit
    f(x, A, λ::Number) = [norm(x, 0, λ); norm(A * x, 2, λ)] # implicit
    return f
end


# Derive the best n linear independent columns of a matrix
function linear_independent_columns(
    A::AbstractMatrix{T},
    rtol::T = convert(T, 0.1),
) where {T}
    iszero(rtol) && return A
    rA = rank(A)
    @static if VERSION < v"1.7.0"
        qr_ = qr(A, Val(true))
    else
        qr_ = qr(A, ColumnNorm())
    end
    r_ = abs.(diag(qr_.R))
    r = findlast(r_ .>= rtol * first(r_))
    r = max(min(r, rA), 1)
    A_ = zeros(T, size(A, 1), r)
    idxs = sort(qr_.p[1:r])
    for i = 1:r
        A_[:, i] .= A[:, idxs[i]]
    end
    return A_
end
