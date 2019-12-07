function hankel(a::AbstractArray, b::AbstractArray) where T <: Number
    p = vcat([a; b[2:end]])
    n = length(b)-1
    m = length(p)-n+1
    H = Array{eltype(p)}(undef, n, m)
    @inbounds for i in 1:n
        @inbounds for j in 1:m
            H[i,j] = p[i+j-1]
        end
    end
    return H
end

function HankelDMD(X::AbstractArray,  n::Int64, m::Int64)
    return HankelDMD(X[:, 1:end-1], X[:, 2:end], n, m)
end

function HankelDMD(X::AbstractArray, Y::AbstractArray, n::Int64 , m::Int64; dt::T1 = 0.0, threshold::T2 = 0.0) where {T1 <: Real, T2 <: Real}
    @assert threshold >= zero(eltype(threshold))
    @assert size(X)[2] >= n+m
    @assert size(X) == size(Y)
    # TODO Assert size a priori
    # Double check the method with examples
    H₀ = [] # Hankel matrices
    H₁ = []

    @inbounds for i in 1:size(X)[1]
        push!(H₀, hankel(@view(X[i, 1:n]), @view(X[i, n:n+m])))
        push!(H₁, hankel(@view(Y[i, 1:n]), @view(Y[i, n:n+m])))
    end

    # Scale with α
    @inbounds for (h0, h1) in zip(H₀, H₁)
        h0 *= norm(h0[:,1])/norm(X[:, 1])
        h1 *= norm(h0[:,1])/norm(X[:, 1])
    end

    # Combine the measurements
    H₀ = vcat(H₀...)
    H₁ = vcat(H₁...)

    return koopman_svd(H₀, H₁, :HankelDMD, dt, size(X)[1], threshold)
end
