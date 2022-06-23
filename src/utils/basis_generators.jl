for f in [
    :chebyshev_basis,
    :sin_basis,
    :cos_basis,
    :fourier_basis,
    :polynomial_basis,
    :monomial_basis,
]
    @eval $f(x, c) = $f(scalarize(x), c)
end
function _generateBasis!(eqs, f, x, coeffs)
    n_x = size(x, 1)
    @assert length(eqs) == size(x, 1) * length(coeffs)
    @inbounds for (i, ti) in enumerate(coeffs)
        eqs[((i - 1) * n_x + 1):(i * n_x)] .= f(x, ti)
    end
    return
end

"""
$(SIGNATURES)

Constructs an array containing a Chebyshev basis in the variables `x` with coefficients `c`.
If `c` is an `Int` returns all coefficients from 1 to `c`.
"""
function chebyshev_basis(x::Array, coefficients::AbstractVector)
    eqs = Array{Num}(undef, size(x, 1) * length(coefficients))
    f(x, t) = cos.(t .* acos.(x))
    _generateBasis!(eqs, f, x, coefficients)
    eqs
end

chebyshev_basis(x::Array, terms::Int) = chebyshev_basis(x, 1:terms)

"""
$(SIGNATURES)

Constructs an array containing a Sine basis in the variables `x` with coefficients `c`.
If `c` is an `Int` returns all coefficients from 1 to `c`.
"""
function sin_basis(x::Array, coefficients::AbstractVector)
    eqs = Array{Num}(undef, size(x, 1) * length(coefficients))
    f(x, t) = sin.(t .* x)
    _generateBasis!(eqs, f, x, coefficients)
    eqs
end

sin_basis(x::Array, terms::Int) = sin_basis(x, 1:terms)

"""
$(SIGNATURES)

Constructs an array containing a Cosine basis in the variables `x` with coefficients `c`.
If `c` is an `Int` returns all coefficients from 1 to `c`.
"""
function cos_basis(x::Array, coefficients::AbstractVector)
    eqs = Array{Num}(undef, size(x, 1) * length(coefficients))
    f(x, t) = cos.(t .* x)
    _generateBasis!(eqs, f, x, coefficients)
    eqs
end

cos_basis(x::Array, terms::Int) = cos_basis(x, 1:terms)

"""
$(SIGNATURES)

Constructs an array containing a Fourier basis in the variables `x` with (integer) coefficients `c`.
If `c` is an `Int` returns all coefficients from 1 to `c`.
"""
function fourier_basis(x::Array, coefficients::AbstractVector{Int})
    eqs = Array{Num}(undef, size(x, 1) * length(coefficients))
    f(x, t) = iseven(t) ? cos.(t .* x ./ 2) : sin.(t .* x ./ 2)
    _generateBasis!(eqs, f, x, coefficients)
    eqs
end

fourier_basis(x::Array, terms::Int) = fourier_basis(x, 1:terms)

"""
$(SIGNATURES)

Constructs an array containing a polynomial basis in the variables `x` up to degree `c` of the form
`[x₁, x₂, x₃, ..., x₁^1 * x₂^(c-1)]`. Mixed terms are included.
"""
function polynomial_basis(x::Array, degree::Int = 1)
    @assert degree > 0
    n_x = length(x)
    n_c = binomial(n_x + degree, degree)
    eqs = Array{Num}(undef, n_c)
    _check_degree(x) = sum(x) <= degree ? true : false
    itr = Base.Iterators.product([0:degree for i in 1:n_x]...)
    itr_ = Base.Iterators.Stateful(Base.Iterators.filter(_check_degree, itr))
    filled = false
    @inbounds for i in 1:n_c
        eqs[i] = 1
        filled = true
        for (xi, ci) in zip(x, popfirst!(itr_))
            if !iszero(ci)
                filled ? eqs[i] = xi^ci : eqs[i] *= xi^ci
                filled = false
            end
        end
    end
    eqs
end

"""
$(SIGNATURES)

Constructs an array containing monomial basis in the variables `x` up to degree `c` of the form
`[x₁, x₁^2, ... , x₁^c, x₂, x₂^2, ...]`.
"""
function monomial_basis(x::AbstractArray, degree::Int = 1)
    @assert degree > 0
    n_x = length(x)
    exponents = 1:degree
    n_e = length(exponents)
    n_c = n_x * n_e + 1
    eqs = Array{Num}(undef, n_c)
    eqs[1] = Num(1)
    idx = 0
    for i in 1:n_x, j in 1:n_e
        idx = (i - 1) * n_e + j + 1
        eqs[idx] = x[i]^exponents[j]
    end
    eqs
end
