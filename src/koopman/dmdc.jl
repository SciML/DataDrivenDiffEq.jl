function DMDc(X::AbstractArray, U::AbstractArray; B::AbstractArray = [], alg::AbstractKoopmanAlgorithm = DMDPINV())
    return DMDc(X[:, 1:end-1], X[:, 2:end], U, B = B, alg = alg)
end

function DMDc(X::AbstractArray, Y::AbstractArray, U::AbstractArray; B::AbstractArray = [], alg::AbstractKoopmanAlgorithm = DMDPINV())
    @assert size(X)[2] .== size(Y)[2] "Provide consistent dimensions for data"
    @assert size(Y)[1] .<= size(Y)[2] "Provide consistent dimensions for data"
    @assert size(X)[2] == size(U)[2] "Provide consistent input data."

    nₓ = size(X)[1]
    nᵤ = size(U)[1]

    Ω = vcat(X, U)

    if isempty(B)
        G = alg(Ω, Y)

        A = G[:, 1:nₓ]
        B = G[:, nₓ+1:end]

    else
        A = alg(X, Y-B*U)
    end

    return LinearKoopman(A, B, Y*Ω', Ω*Ω', true)
end

function gDMDc(X::AbstractArray, Y::AbstractArray, U::AbstractArray; B::AbstractArray = [], alg::AbstractKoopmanAlgorithm = DMDPINV())
    @assert size(X)[2] .== size(Y)[2] "Provide consistent dimensions for data"
    @assert size(Y)[1] .<= size(Y)[2] "Provide consistent dimensions for data"
    @assert size(X)[2]-1 == size(U)[2] "Provide consistent input data."

    nₓ = size(X)[1]
    nᵤ = size(U)[1]

    Ω = vcat(X, U)

    if isempty(B)
        G = alg(Ω, Y)

        A = G[:, 1:nₓ]
        B = G[:, nₓ+1:end]

    else
        A = alg(X, Y-B*U)
    end


    return LinearKoopman(A, B, Y*Ω', Ω*Ω', false)
end
