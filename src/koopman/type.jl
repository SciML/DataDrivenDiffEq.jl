"""
$(TYPEDEF)

A special basis over the states with parameters , independent variable  and possible exogenous controls.
It extends an `AbstractBasis`, which also stores information about the lifted dynamics, specified by a sufficient
matrix factorization, an output mapping and internal variables to update the equations. It can be called with the typical SciML signature, meaning out of place with `f(u,p,t)`
or in place with `f(du, u, p, t)`. If control inputs are present, it is assumed that no control corresponds to
zero for all inputs. The corresponding function calls are `f(u,p,t,inputs)` and `f(du,u,p,t,inputs)` and need to
be specified fully.

If `linear_independent` is set to `true`, a linear independent basis is created from all atom functions in `f`.

If `simplify_eqs` is set to `true`, `simplify` is called on `f`.

Additional keyworded arguments include `name`, which can be used to name the basis, and
`observed` for defining observeables.



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
mutable struct Koopman{O,M,G,T} <: AbstractKoopman
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
    """Internal function representation of the basis"""
    f::Function
    """Associated lifting of the operator"""
    lift::Function
    """Name of the basis"""
    name::Symbol
    """Internal systems"""
    systems::Vector{Basis}
    """Discrete or time continuous"""
    is_discrete::Bool
    """The operator/generator of the dynamics"""
    K::O
    """Mapping back onto the observed states"""
    C::M
    """Internal matrix `Q` used for updating"""
    Q::G
    """Internal matrix `P` used for updating"""
    P::T
end

function Koopman(eqs::AbstractVector{Equation}, states::AbstractVector;
    K::O = diagm(ones(eltype(states), length(eqs))),
    C::AbstractMatrix = diagm(ones(eltype(K), length(eqs))),
    Q::AbstractMatrix = zeros(eltype(states), 0,0),
    P::AbstractMatrix = zeros(eltype(states), 0,0),
    lift::Function = (X, args...)->identity(X),
    parameters::AbstractVector = [], iv = nothing,
    controls::AbstractVector = [], observed::AbstractVector = [],
    name = gensym(:Koopman), is_discrete::Bool = true,
    digits::Int = 10, # Round all coefficients
    simplify = false, linear_independent = false,
    eval_expression = false,
    kwargs...) where O <: Union{AbstractMatrix, Eigen, Factorization}

    iv === nothing && (iv = Variable(:t))
    iv = value(iv)
    eqs = scalarize(eqs)
    states, controls, parameters, observed = value.(scalarize(states)), value.(scalarize(controls)), value.(scalarize(parameters)), value.(scalarize(observed))

    eqs = [eq for eq in eqs if ~isequal(Num(eq),zero(Num))]

    lhs = Num[x.lhs for x in eqs]
    eqs_ = Num[x.rhs for x in eqs]

    if linear_independent
        eqs_ = create_linear_independent_eqs(eqs_, simplify)
    else
        eqs_ = simplify ? ModelingToolkit.simplify.(eqs_) : eqs_
    end

    unique!(eqs_, !simplify)

    f = DataDrivenDiffEq._build_ddd_function(eqs_, states, parameters, iv, controls, eval_expression)

    eqs = [lhs[i] ~ eq for (i,eq) ∈ enumerate(eqs_)]

    return Koopman{typeof(K), typeof(C), typeof(Q), typeof(P)}(eqs,
    states, controls, parameters, observed , iv, f, lift, name, Basis[],
    is_discrete, K, C, Q, P)
end

function Koopman(eqs::AbstractVector{Num}, states::AbstractVector;
    K::O = diagm(ones(eltype(states), length(eqs))),
    C::AbstractMatrix = diagm(ones(eltype(K), length(eqs))),
    Q::AbstractMatrix = zeros(eltype(states), 0,0),
    P::AbstractMatrix = zeros(eltype(states), 0,0),
    lift::Function = (X, args...)->identity(X),
    parameters::AbstractVector = [], iv = nothing,
    controls::AbstractVector = [], observed::AbstractVector = [],
    name = gensym(:Koopman), is_discrete::Bool = true,
    digits::Int = 10, # Round all coefficients
    simplify = false, linear_independent = false,
    eval_expression = false,
    kwargs...) where O <: Union{AbstractMatrix, Eigen, Factorization}

    iv === nothing && (iv = Variable(:t))
    iv = value(iv)
    eqs = scalarize(eqs)
    states, controls, parameters, observed = value.(scalarize(states)), value.(scalarize(controls)), value.(scalarize(parameters)), value.(scalarize(observed))

    eqs_ = [eq for eq in eqs if ~isequal(Num(eq),zero(Num))]

    if linear_independent
        eqs_ = create_linear_independent_eqs(eqs_, simplify)
    else
        eqs_ = simplify ? ModelingToolkit.simplify.(eqs_) : eqs_
    end

    unique!(eqs_, !simplify)

    f = DataDrivenDiffEq._build_ddd_function(eqs_, states, parameters, iv, controls, eval_expression)
    D = Differential(iv)

    eqs = [D(states[i]) ~ eq for (i,eq) ∈ enumerate(eqs_)]

    return Koopman{typeof(K), typeof(C), typeof(Q), typeof(P)}(eqs,
    states, controls, parameters, observed, iv, f, lift, name, Basis[],
    is_discrete, K, C, Q, P)
end



# We assume that we only have real valued observed
Base.Matrix(k::AbstractKoopman) = real.(Matrix(_get_K(k)))

# Get the lifting function
lifting(k::AbstractKoopman) = getfield(k, :lift)

# Get K
_get_K(k::AbstractKoopman) = getfield(k, :K)

"""
$(SIGNATURES)

Returns `true` if the `AbstractKoopmanOperator` `k` is discrete in time.
"""
is_discrete(k::AbstractKoopman) = getfield(k, :is_discrete)



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
    K = _get_K(k)
    isa(K, Eigen) && return K 
    eigen(K)
end

"""
$(SIGNATURES)

Return the eigenvalues of the `AbstractKoopmanOperator`.
"""
LinearAlgebra.eigvals(k::AbstractKoopman) = eigvals(_get_K(k))

"""
$(SIGNATURES)

Return the eigenvectors of the `AbstractKoopmanOperator`.
"""
LinearAlgebra.eigvecs(k::AbstractKoopman) = eigvecs(_get_K(k))

"""
$(SIGNATURES)

Return the eigenvectors of a continuous `AbstractKoopmanOperator`.
"""
modes(k::AbstractKoopman) = is_continuous(k) ? eigvecs(k) : throw(AssertionError("Koopman is discrete."))

"""
$(SIGNATURES)

Return the eigenvalues of a continuous `AbstractKoopmanOperator`.
"""
frequencies(k::AbstractKoopman) = is_continuous(k) ? eigvals(k) : throw(AssertionError("Koopman is discrete."))

"""
$(SIGNATURES)

Return the approximation of the discrete Koopman operator stored in `k`.
"""
operator(k::AbstractKoopman) = is_discrete(k) ? _get_K(k) : throw(AssertionError("Koopman is continouos."))

"""
$(SIGNATURES)

Return the approximation of the continuous Koopman generator stored in `k`.
"""
generator(k::AbstractKoopman) = is_continuous(k) ? _get_K(k) : throw(AssertionError("Koopman is discrete."))

"""
$(SIGNATURES)

Return the array `C`, mapping the Koopman space back onto the state space.
"""
outputmap(k::AbstractKoopman) = k.C

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
is_stable(k::AbstractKoopman) = begin 
    K = _get_K(k)
    is_discrete(k) && all(real.(eigvals(k)) .< real.(one(eltype(K)))) 
    all(real.(eigvals(k)) .< zero(eltype(K)))
end

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

    ϵ = norm(Ψ₁-Matrix(k)*Ψ₀, 2)

    if ϵ < threshold
        return
    end

    k.Q += Ψ₁*Ψ₀'
    k.P += Ψ₀*Ψ₀'
    k.operator .= k.Q / k.P

    if norm(Y - outputmap(k)*Matrix(k)*Ψ₀) < threshold
        return
    end

    # TODO Make this a proper rank 1 update
    k.output .= X / Ψ₀

    return
end
