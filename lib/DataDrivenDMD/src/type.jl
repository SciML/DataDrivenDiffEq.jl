"""
$(TYPEDEF)
using ModelingToolkit: get_states
using ModelingToolkit: get_eqs
using ModelingToolkit: get_ctrls
using ModelingToolkit: get_systems
using Base: promote_eltype

A special [`DataDrivenDiffEq.Basis`](@ref) used to represent the Koopman operator and its embedding.


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
struct Koopman{I, D, O, T} <: AbstractKoopman{I}
    """The equations of the basis"""
    eqs::Vector{Equation}
    """Dependent (state) variables"""
    states::Vector
    """Control variables"""
    ctrls::Vector
    """Parameters"""
    ps::Vector
    """Observed"""
    observed::Vector
    """Independent variable"""
    iv::Num
    """Implicit variables of the basis"""
    implicit::Vector
    """Internal function representation of the basis"""
    f::Function
    """Name of the basis"""
    name::Symbol
    """Internal systems"""
    systems::Vector{AbstractBasis}
    """The operator/generator of the dynamics"""
    K::O
    """Mapping back onto the observed states"""
    C::AbstractMatrix{T}
    """Internal matrix `Q` used for updating"""
    Q::AbstractMatrix{T}
    """Internal matrix `P` used for updating"""
    P::AbstractMatrix{T}

    function Koopman(eqs, states, ctrls, ps, observed, iv, implicit, f, name, systems, K, C,
                     Q, P; is_discrete::Bool = true,
                     checks::Bool = true)
        if checks
            # Currently do nothing here
            #check_variables(dvs, iv)
            #check_parameters(ps, iv)
            #check_equations(deqs, iv)
            #check_equations(equations(events), iv)
            #all_dimensionless([dvs; ps; iv]) || check_units(deqs)
        end

        imp_ = !isempty(implicit)
        ctype = Base.promote_eltype(C, Q, P)
        return new{imp_, is_discrete, typeof(K), ctype}(eqs, states, ctrls, ps, observed,
                                                        iv, implicit, f, name, systems,
                                                        K, C, Q, P)
    end
end

function Koopman(eqs::AbstractVector, states::AbstractVector;
                 parameters::AbstractVector = [], iv = nothing,
                 controls::AbstractVector = [], implicits = [],
                 observed::AbstractVector = [],
                 eval_expression = false,
                 K::O = diagm(ones(Float64, length(eqs))),
                 C::AbstractMatrix = diagm(ones(Float64, length(eqs))),
                 Q::AbstractMatrix = zeros(Float64, 0, 0),
                 P::AbstractMatrix = zeros(Float64, 0, 0),
                 simplify = false, linear_independent = false,
                 is_discrete::Bool = true,
                 name = is_discrete ? gensym(:KoopmanOperator) : gensym(:KoopmanGenerator),
                 kwargs...) where {O <: Union{AbstractMatrix, Eigen, Factorization}}
    args_ = DataDrivenDiffEq.__preprocess_basis(eqs, states, controls, parameters, observed,
                                                iv,
                                                implicits, name, AbstractBasis[], simplify,
                                                linear_independent, eval_expression)

    return Koopman(args_..., K, C, Q, P; is_discrete = is_discrete)
end

Base.eltype(k::Koopman{<:Any, <:Any, <:Any, T}) where {T} = T

## Koopman methods

# We assume that we only have real valued observed
Base.Matrix(k::AbstractKoopman) = real.(Matrix(__get_K(k)))

# Get K
__get_K(k::AbstractKoopman) = getfield(k, :K)

"""
$(SIGNATURES)

Returns `true` if the `AbstractKoopmanOperator` `k` is discrete in time.
"""
is_discrete(k::Koopman{<:Any, D}) where {D} = D

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
modes(k::Koopman{<:Any, true}) = throw(AssertionError("Koopman is discrete."))
modes(k::Koopman{<:Any, false}) = eigvecs(k)

"""
$(SIGNATURES)

Return the eigenvalues of a continuous `AbstractKoopmanOperator`.
"""
frequencies(k::Koopman{<:Any, false}) = eigvals(k)
frequencies(k::Koopman{<:Any, true}) = throw(AssertionError("Koopman is discrete."))

"""
$(SIGNATURES)

Return the approximation of the discrete Koopman operator stored in `k`.
"""
operator(k::Koopman{<:Any, true}) = __get_K(k)
operator(k::Koopman{<:Any, false}) = throw(AssertionError("Koopman is continouos."))

"""_
$(SIGNATURES)

Return the approximation of the continuous Koopman generator stored in `k`.
"""
generator(k::Koopman{<:Any, false}) = __get_K(k)
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

+ the Koopman operator has just eigenvalues with magnitude less than one or
+ the Koopman generator has just eigenvalues with a negative real part
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
