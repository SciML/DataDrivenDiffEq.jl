mutable struct Koopman{O,M,G,T} <: AbstractKoopman
    """The equations of the basis"""
    eqs::Vector{Equation}
    """Dependent (state) variables"""
    states::Vector
    """Control variables"""
    controls::Vector
    """Parameters"""
    ps::Vector
    """Observed"""
    observed::Vector
    """Independent variable"""
    iv::Num
    """Internal function representation of the basis"""
    f::Function
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

function _round!(x::AbstractArray{T, N}, digits::Int) where {T, N}
    for i in eachindex(x)
        x[i] = round(x[i], digits = digits)
    end
    return x
end

function Koopman(eqs::AbstractVector{Num}, states::AbstractVector{Num};
    K::O = diagm(ones(eltype(states), length(eqs))),
    C::AbstractMatrix = diagm(ones(eltype(K), length(eqs))),
    Q::AbstractMatrix = zeros(eltype(states), 0,0),
    P::AbstractMatrix = zeros(eltype(states), 0,0),
    B::AbstractMatrix = zeros(eltype(states), 0, 0),
    parameters::AbstractVector = [], iv = nothing,
    controls::AbstractVector = [], observed::AbstractVector = [],
    name = gensym(:Koopman), is_discrete::Bool = true,
    digits::Int = 10, # Round all coefficients
    simplify = false, linear_independent = false,
    eval_expression = false, s_idxs::BitVector = BitVector(), # Indicator for state eqs
    kwargs...) where O <: Union{AbstractMatrix, Eigen, Factorization}

    s_idxs = isempty(s_idxs) ? [true for i in 1:length(eqs)] : s_idxs

    eqs_ = _round!(real.(C[:,s_idxs]*Matrix(K)), digits)*eqs[s_idxs]

    if !isempty(B)
        eqs_ += _round!(C[:, s_idxs]*B, digits)*eqs[.! s_idxs]
    end

    if linear_independent
        eqs_ = create_linear_independent_eqs(eqs_, simplify)
    else
        eqs_ = simplify ? ModelingToolkit.simplify.(eqs_) : eqs_
    end

    isnothing(iv) && (iv = Num(Variable(:t)))
    unique!(eqs_, !simplify)

    f = DataDrivenDiffEq._build_ddd_function(eqs_, states, parameters, iv, controls, eval_expression)

    D = Differential(iv)
    eqs = [D(states[i]) ~ eq for (i,eq) ∈ enumerate(eqs_)]

    return Koopman{typeof(K), typeof(C), typeof(Q), typeof(P)}(eqs,
    value.(states), value.(controls), value.(parameters), value.(observed), value(iv), f, name, Basis[],
    is_discrete, K, C, Q, P)
end


#function Koopman(K::AbstractMatrix{T};
#    B::AbstractMatrix = zeros(eltype(T), 0, 0), kwargs...) where T <: Real
#    n, m = size(K)
#    n_c, m_c = size(B)
#
#    if m_c > 0
#        @variables x[1:n] u[1:m_c]
#        Koopman([x; u], x, controls = u, K = K ,B = B,
#            s_idxs = BitVector([i <= n ? true : false for i in 1:n+m_c]),
#            kwargs...)
#    else
#        @variables x[1:n]
#        Koopman(x, x, K = K, kwargs...)
#    end
#end
#
#function Koopman(K::Eigen;
#    B::AbstractMatrix = zeros(eltype(T), 0, 0), kwargs...) where T <: Real
#    n = length(K.values)
#    n_c, m_c = size(B)
#    if m_c > 0
#        @variables x[1:n] u[1:m_c]
#        Koopman([x; u], x, controls = u, K = K ,B = B,
#            s_idxs = BitVector([i <= n ? true : false for i in 1:n+m_c]),
#            kwargs...)
#    else
#        @variables x[1:n]
#        Koopman(x, x, K = K, kwargs...)
#    end
#end



# We assume that we only have real valued observed
Base.Matrix(k::AbstractKoopman) = real.(Matrix(k.K))

"""
$(SIGNATURES)

Returns if the `AbstractKoopmanOperator` `k` is discrete in time.
"""
is_discrete(k::AbstractKoopman) = k.is_discrete

"""
$(SIGNATURES)

Returns if the `AbstractKoopmanOperator` `k` is continuous in time.
"""
is_continuous(k::AbstractKoopman) = !k.is_discrete

"""
$(SIGNATURES)

Return the eigendecomposition of the `AbstractKoopmanOperator`.
"""
LinearAlgebra.eigen(k::AbstractKoopman) = isa(k.K, Eigen) ? k.K : eigen(k.K)

"""
$(SIGNATURES)

Return the eigenvalues of the `AbstractKoopmanOperator`.
"""
LinearAlgebra.eigvals(k::AbstractKoopman) = eigvals(k.K)

"""
$(SIGNATURES)

Return the eigenvectors of the `AbstractKoopmanOperator`.
"""
LinearAlgebra.eigvecs(k::AbstractKoopman) = eigvecs(k.K)

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
operator(k::AbstractKoopman) = is_discrete(k) ? k.K : throw(AssertionError("Koopman is continouos."))

"""
$(SIGNATURES)

Return the approximation of the continuous Koopman generator stored in `k`.
"""
generator(k::AbstractKoopman) = is_continuous(k) ? k.K : throw(AssertionError("Koopman is discrete."))

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
is_stable(k::AbstractKoopman) = is_discrete(k) ? all(real.(eigvals(k)) .< real.(one(eltype(k.K)))) : all(real.(eigvals(k)) .< zero(eltype(k.K)))

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

#"""
#    reduce_basis(k; threshold)
#
#Reduces the `basis` of the nonlinear Koopman using the 1-norm of each row
#of the matrix `C*K`. Rows where the threshold is not reached are deleted.
#"""
#function reduce_basis(k::AbstractKoopman; threshold = 1e-5, kwargs...)
#    b = k.output*k.operator
#    inds = vec(sum(abs, b, dims = 1) .> threshold)
#    return Basis(map(x->x.rhs, k.basis[inds]), variables(k.basis), parameters = parameters(k.basis), iv = independent_variable(k.basis), kwargs...)
#end
