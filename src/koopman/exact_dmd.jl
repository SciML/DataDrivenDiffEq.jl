function DMD(X::AbstractArray; alg::AbstractKoopmanAlgorithm = DMDPINV())
    return DMD(X[:, 1:end-1], X[:, 2:end], alg = alg)
end

function DMD(X::AbstractArray, Y::AbstractArray; alg::AbstractKoopmanAlgorithm = DMDPINV())
    @assert size(X)[2] .== size(Y)[2] "Provide consistent dimensions for data"
    @assert size(Y)[1] .<= size(Y)[2] "Provide consistent dimensions for data"

    # Best Frob norm approximator
    A = alg(X, Y)

    return LinearKoopman(A, zero(eltype(A))*I(size(A,1)), Y*X', X*X', true)
end

function gDMD(X::AbstractArray, Y::AbstractArray; alg::AbstractKoopmanAlgorithm = DMDPINV())
    @assert size(X)[2] .== size(Y)[2] "Provide consistent dimensions for data"
    @assert size(Y)[1] .<= size(Y)[2] "Provide consistent dimensions for data"

    # Best Frob norm approximator
    A = alg(X, Y)

    return LinearKoopman(A, zero(eltype(A))*I(size(A,1)), Y*X', X*X', false)
end
