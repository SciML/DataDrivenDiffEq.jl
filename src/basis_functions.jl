
#mutable struct BasisFunction
#    s::String # String representation
#    f::Function # Function
#
#    function BasisFunction(s::String)
#        f = eval(Meta.parse("x -> "*s))
#        return new(s,f)
#    end
#end
#
#(b::BasisFunction)(x::AbstractArray) = b.f(x)
#
#mutable struct BasisCandidate
#    basis::Array{BasisFunction}
#
#    function BasisCandidate()
#        return new(Array{BasisFunction}(undef, 0))
#    end
#end
#
#
#function BasisCandidate(s::Array{String})
#    c = BasisCandidate()
#    @inbounds for s_ in s
#        push!(c, BasisFunction(s_))
#    end
#    return c
#end
#
#function BasisCandidate(b::Array{BasisFunction})
#    c = BasisCandidate()
#    @inbounds for b_ in b
#        push!(c, b_)
#    end
#    return c
#end
#
#
## Make the struct callable
#function (b::BasisCandidate)(x::AbstractArray)
#    if !isempty(b.basis)
#        return vcat([bi(x) for bi in b.basis])
#    else
#        # Identity
#        return x
#    end
#end
#
#Base.push!(c::BasisCandidate, b::BasisFunction) = push!(c.basis, b)
#
#function Base.deleteat!(b::BasisCandidate, i::Int64)
#    deleteat!(b.basis, i)
#end
#
#Base.size(c::BasisCandidate) = size(c.basis)
#
#function evaluate(c::BasisCandidate, X::Array)
#    nₓ, nₘ = size(X)
#    m = size(c)[1]
#    Y = Array{eltype(X)}(undef, (m, nₘ))
#    evaluate!(Y, c, X)
#    return Y
#end
#
#function evaluate!(Y::Array, c::BasisCandidate, X::Array)
#    nₓ, nₘ = size(X)
#    m = size(c)[1]
#    @assert all(size(Y) .== (m, nₘ))
#    @inbounds for i in 1:nₘ
#        Y[:, i] = c(view(X, :,i))
#    end
#end
#
#function collapse(b::BasisCandidate, Ξ::AbstractArray; threshold::Float64 = 1e-3)
#    @assert length(b.basis) == size(Ξ, 2)
#    inds = sum(abs.(Ξ), dims = 1) .> threshold
#    return BasisCandidate(b.basis[vec(inds)]), Ξ[:, vec(inds)]
#end
#
