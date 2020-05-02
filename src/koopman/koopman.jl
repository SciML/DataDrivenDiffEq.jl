"""
    is_discete(k)

Returns, if the `AbstractKoopmanOperator` `k` is discrete in time.
"""
is_discrete(k::AbstractKoopmanOperator) = k.discrete

"""
    is_continouos(k)

Returns, if the `AbstractKoopmanOperator` `k` is continouos in time.
"""
is_continouos(k::AbstractKoopmanOperator) = !k.discrete

"""
    eigen(k)

    Return the eigendecomposition of the `AbstractKoopmanOperator`.
"""
LinearAlgebra.eigen(k::AbstractKoopmanOperator) = eigen(k.operator)

"""
    eigevals(k)

    Return the eigenvalues of the `AbstractKoopmanOperator`.
"""
LinearAlgebra.eigvals(k::AbstractKoopmanOperator) = eigvals(k.operator)

"""
    eigvecs(k)

    Return the eigenvectors of the `AbstractKoopmanOperator`.
"""
LinearAlgebra.eigvecs(k::AbstractKoopmanOperator) = eigvecs(k.operator)

"""
    modes(k)

    Return the eigenvectors of a continuous `AbstractKoopmanOperator`.
"""
modes(k::AbstractKoopmanOperator) = is_continouos(k) ? eigvecs(k) : throw(AssertionError("Koopman is discrete."))

"""
    frequencies(k)

    Return the eigenvalues of a continuous `AbstractKoopmanOperator`.
"""
frequencies(k::AbstractKoopmanOperator) = is_continouos(k) ? eigvals(k) : throw(AssertionError("Koopman is discrete."))

"""
    operator(k)

    Return the approximation of the discrete koopman operator stored in `k`.
"""
operator(k::AbstractKoopmanOperator) = is_discrete(k) ? k.operator : throw(AssertionError("Koopman is continouos."))

"""
    generator(k)

    Return the approximation of the continuous koopman generator stored in `k`.
"""
generator(k::AbstractKoopmanOperator) = is_continouos(k) ? k.operator : throw(AssertionError("Koopman is discrete."))

"""
    inputmap(k)

    Return the array `B`, mapping the exogenuos inputs to the koopman space.
"""
inputmap(k::AbstractKoopmanOperator) = k.input

"""
    outputmap(k)

    Return the array `C`, mapping the koopman space back onto the state space.
"""
outputmap(k::AbstractKoopmanOperator) = k.output

"""
    updateable(k)

    Returns `true` if the `AbstractKoopmanOperator` is updateable.
"""
updateable(k::AbstractKoopmanOperator) = !isempty(k.Q) && !isempty(k.P)

"""
    isstable(k)

    Returns `true` if either

    + the koopman operator has just eigenvalues with magnitude less than one or
    + the koopman generator has just eigenvalues with a negative real part
"""
isstable(k::AbstractKoopmanOperator) = is_discrete(k) ? all(abs.(eigvals(k)) .< one(eltype(k.operator))) : all(real.(eigvals(k)) < zero(eltype(k.operator)))
