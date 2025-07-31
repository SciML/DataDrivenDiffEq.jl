"""
$(TYPEDEF)

# Fields

$(FIELDS)

## Note

The keyword argument `eval_expression` controls the function creation
behavior. `eval_expression=true` means that `eval` is used, so normal
world-age behavior applies (i.e. the functions cannot be called from
the function that generates them). If `eval_expression=false`,
then construction via GeneralizedGenerated.jl is utilized to allow for
same world-age evaluation. However, this can cause Julia to segfault
on sufficiently large basis functions. By default eval_expression=false.
"""
struct Koopman{T, B <: AbstractBasis, K, DISCRETE} <: AbstractKoopman
    """The basis of observables"""
    basis::B
    """The operator/generator of the dynamics"""
    K::K
    """Mapping back onto the observed states"""
    C::AbstractMatrix{T}
    """Internal matrix `Q` used for updating"""
    Q::AbstractMatrix{T}
    """Internal matrix `P` used for updating"""
    P::AbstractMatrix{T}
end

Base.eltype(k::Koopman{T}) where {T} = T

## Koopman methods

# We assume that we only have real valued observed
Base.Matrix(k::AbstractKoopman) = real.(Matrix(__get_K(k)))

# Get K
__get_K(k::AbstractKoopman) = getfield(k, :K)

"""
$(SIGNATURES)

Returns `true` if the `AbstractKoopmanOperator` `k` is discrete in time.
"""
is_discrete(k::Koopman{<:Any, <:Any, <:Any, D}) where {D} = D

"""
$(SIGNATURES)

Returns `true` if the `AbstractKoopmanOperator` `k` is continuous in time.
"""
is_continuous(k::AbstractKoopman) = !is_discrete(k)

"""
$(SIGNATURES)

Return the eigendecomposition of the `AbstractKoopmanOperator`.
"""
LinearAlgebra.eigen(k::AbstractKoopman) = begin
    K = __get_K(k)
    isa(K, Eigen) && return K
    eigen(K)
end

"""
$(SIGNATURES)

Return the eigenvalues of the `AbstractKoopmanOperator`.
"""
LinearAlgebra.eigvals(k::AbstractKoopman) = eigvals(__get_K(k))

"""
$(SIGNATURES)

Return the eigenvectors of the `AbstractKoopmanOperator`.
"""
LinearAlgebra.eigvecs(k::AbstractKoopman) = eigvecs(__get_K(k))

"""
$(SIGNATURES)

Return the eigenvectors of a continuous `AbstractKoopmanOperator`.
"""
modes(k::Koopman{<:Any, <:Any, <:Any, true}) = throw(AssertionError("Koopman is discrete."))
modes(k::Koopman{<:Any, <:Any, <:Any, false}) = eigvecs(k)

"""
$(SIGNATURES)

Return the eigenvalues of a continuous `AbstractKoopmanOperator`.
"""
frequencies(k::Koopman{<:Any, <:Any, <:Any, false}) = eigvals(k)
function frequencies(k::Koopman{<:Any, <:Any, <:Any, true})
    throw(AssertionError("Koopman is discrete."))
end

"""
$(SIGNATURES)

Return the approximation of the discrete Koopman operator stored in `k`.
"""
operator(k::Koopman{<:Any, <:Any, <:Any, true}) = __get_K(k)
function operator(k::Koopman{<:Any, <:Any, <:Any, false})
    throw(AssertionError("Koopman is continuous."))
end

"""
_
$(SIGNATURES)

Return the approximation of the continuous Koopman generator stored in `k`.
"""
generator(k::Koopman{<:Any, <:Any, <:Any}) = __get_K(k)
generator(k::Koopman{<:Any, true}) = throw(AssertionError("Koopman is discrete."))

"""
$(SIGNATURES)

Return the array `C`, mapping the Koopman space back onto the state space.
"""
outputmap(k::AbstractKoopman) = Symbolics.unwrap(getfield(k, :C))

"""
$(SIGNATURES)

Returns `true` if the `AbstractKoopmanOperator` is updatable.
"""
updatable(k::AbstractKoopman) = !isempty(k.Q) && !isempty(k.P)

"""
$(SIGNATURES)

Returns `true` if either:

  - the Koopman operator has just eigenvalues with magnitude less than one or
  - the Koopman generator has just eigenvalues with a negative real part
"""
is_stable(k::Koopman{<:Any, true}) = all(real.(eigvals(k)) .< real.(one(eltype(k))))
is_stable(k::Koopman{<:Any, false}) = all(real.(eigvals(k)) .< real.(zero(eltype(k))))

# TODO This does not work, since we are using the reduced basis instead of the
# original, lifted dynamics...
"""
$(SIGNATURES)

Update the Koopman `k` given new data `X` and `Y`. The operator is updated in place if
the L2 error of the prediction exceeds the `threshold`.

`p` and `t` are the parameters of the basis and the vector of timepoints, if necessary.
"""
function update!(k::AbstractKoopman,
        X::AbstractArray, Y::AbstractArray;
        p::AbstractArray = [], t::AbstractVector = [],
        U::AbstractArray = [],
        threshold::T = eps()) where {T <: Real}
    @assert updatable(k) "Linear Koopman is not updatable."

    Ψ₀ = k(X, p, t, U)
    Ψ₁ = k(Y, p, t, U)

    ϵ = norm(Ψ₁ - Matrix(k) * Ψ₀, 2)

    if ϵ < threshold
        return
    end

    k.Q += Ψ₁ * Ψ₀'
    k.P += Ψ₀ * Ψ₀'
    k.operator .= k.Q / k.P

    if norm(Y - outputmap(k) * Matrix(k) * Ψ₀) < threshold
        return
    end

    # TODO Make this a proper rank 1 update
    k.output .= X / Ψ₀

    return
end
