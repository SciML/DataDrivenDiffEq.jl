using LinearAlgebra
using DataDrivenDiffEq
using DelayEmbeddings

function taken(A, m)
        if m == 0
                return A
        end

        x,y = size(A)

        H = Array{eltype(A)}(undef, x*(m+1), y-m)

        for i in 1:m+1
                H[1+(i-1)*x:i*x, :] = A[:,i:y-(m-i)-1]
        end
        return H
end

A = rand(3,100)

H = taken(A[:, 1:end-1], 10)
H2 = taken(A[:, 2:end],10)

H*pinv(H2)
