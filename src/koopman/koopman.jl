"""
    is_discrete(k)

Returns if the `AbstractKoopmanOperator` `k` is discrete in time.
"""
is_discrete(k::AbstractKoopmanOperator) = k.discrete

"""
    is_continuous(k)

Returns if the `AbstractKoopmanOperator` `k` is continuous in time.
"""
is_continuous(k::AbstractKoopmanOperator) = !k.discrete

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
modes(k::AbstractKoopmanOperator) = is_continuous(k) ? eigvecs(k) : throw(AssertionError("Koopman is discrete."))

"""
    frequencies(k)

    Return the eigenvalues of a continuous `AbstractKoopmanOperator`.
"""
frequencies(k::AbstractKoopmanOperator) = is_continuous(k) ? eigvals(k) : throw(AssertionError("Koopman is discrete."))

"""
    operator(k)

    Return the approximation of the discrete Koopman operator stored in `k`.
"""
operator(k::AbstractKoopmanOperator) = is_discrete(k) ? k.operator : throw(AssertionError("Koopman is continouos."))

"""
    generator(k)

    Return the approximation of the continuous Koopman generator stored in `k`.
"""
generator(k::AbstractKoopmanOperator) = is_continuous(k) ? k.operator : throw(AssertionError("Koopman is discrete."))

"""
    inputmap(k)

    Return the array `B`, mapping the exogenous inputs to the Koopman space.
"""
inputmap(k::AbstractKoopmanOperator) = k.input

"""
    outputmap(k)

    Return the array `C`, mapping the Koopman space back onto the state space.
"""
outputmap(k::AbstractKoopmanOperator) = k.output

"""
    updatable(k)

    Returns `true` if the `AbstractKoopmanOperator` is updatable.
"""
updatable(k::AbstractKoopmanOperator) = !isempty(k.Q) && !isempty(k.P)

"""
    isstable(k)

    Returns `true` if either:

    + the Koopman operator has just eigenvalues with magnitude less than one or
    + the Koopman generator has just eigenvalues with a negative real part
"""
isstable(k::AbstractKoopmanOperator) = is_discrete(k) ? all(abs.(eigvals(k)) .< one(eltype(k.operator))) : all(real.(eigvals(k)) .< zero(eltype(k.operator)))

# For use with LowRankApprox
# Need to find a better way
function LinearAlgebra.eigvals(l::LinearOperator)
    eigvals(Matrix(l))
end

function LinearAlgebra.eigvecs(l::LinearOperator)
    eigvecs(Matrix(l))
end
