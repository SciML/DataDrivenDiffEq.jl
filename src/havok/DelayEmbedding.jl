@with_kw mutable struct DelayEmbedding{R}
    Henkel::Array{R,2} = [0.0 0.0]
    Eigenmodes::Array{R,2} = [0.0 0.0]
    Eigenvalues::Array{R,1} = [0.0]
    Eigenseries::Array{R,2} = [0.0 0.0]
end

function embed(timeseries::AbstractArray; q::Int=0, r::Int=0, d::AbstractFloat=0.0)
    m = length(timeseries)

    print_info(d,q,r)

    if r == 0
        if q == 0
            # WIP
            error("WIP. No parameter chosen.")
        else # Only q especified
            H = HankelMatrix(timeseries, q)     # Henkel Matrix
            F = svd(H')                         # SVD fact object
            r = RankEstimation(F.S, size(H))    # Optimal Hard Threshold (Gavish M., Donoho L., 2014)
            println("Using optimal SVHT a $r-rank projection was obtained.")
        end

    elseif q == 0 # Only r is especified
            # Solve for minimum q ∈ ℤ such that f(q) == r
            q = minimum_q_with_r(timeseries, r)
            println("Minimum bound for delays using optimal SVHT is q=$q")

    else # Both parameters are especified
        println("No automatic parameter selection.")
    end

    # Henkel Matrix & SVD fact object / could be optimized
    H = HankelMatrix(timeseries, q)
    F = svd(H')
    println("Energy retained in projection: $(sum(F.S[1:r])/sum(F.S))")

    return q, r, DelayEmbedding(Henkel=H, Eigenmodes=F.U[:,1:r], Eigenvalues=F.S[1:r], Eigenseries=F.V[:,1:r])
end

# Henkel matrix: q-lagged phase space representation of discrete data. Generalizes traditional HankelMatrix
function HankelMatrix(timeseries::AbstractArray, q::Int)
    m = size(timeseries, 1)
    n = size(timeseries, 2)
    A = zeros(m-q, n*q)
    for i in 1:q
        if n == 1
            A[:,i] = timeseries[i:m-q+i-1]
        else
            for j in 1:n
                A[:,(i-1)*n+j] = timeseries[i:m-q+i-1,j]
            end
        end
    end
    A
end

# Minimum q given r
function minimum_q_with_r(timeseries::AbstractArray, r::Int)
    # Binary search for qlo; f(qlo) == rlo == r
    qhi = r*20
    qlo = r
    rhi = r_from_q(timeseries, qhi)
    rlo = r_from_q(timeseries, qlo)

    # PROBABLY COULD BE OPTIMIZED
    while !(rlo == r) && abs(qlo-qhi)>1
        #map(println,[qlo,qhi])
        qmed = Int(ceil((qhi + qlo)/2))
        rmed = r_from_q(timeseries, qmed)
        if rmed >= r
            qhi = qmed
            rhi = rmed
        else
            qlo = qmed
            rlo = rmed
        end
    end

    # Enforce constraint of minimum q
    while rlo == r
        qlo = qlo - 1
        rlo = r_from_q(timeseries, qlo)
    end

    return qlo + 1
end

# Optimal Hard Threshold Pruning for Singular Values of Hankel Matrix given q
function r_from_q(timeseries::AbstractArray, q::Int)
    H = HankelMatrix(timeseries, q)     # Henkel Matrix
    F = svd(H')                         # SVD fact
    return RankEstimation(F.S, size(H)) # RankEstimation
end


# Rank estimation controller of Gavish M., Donoho L., 2014 methods
function RankEstimation(S::Array{<:AbstractFloat,1}, size::Tuple{Int,Int})
    n, m = size
    #β = m/n
    τ = optimal_svht(m, n; known_noise=false) * median(S) * sqrt(m)
    r = length(findall(x->x.>τ, S))
    r > 0 ? (return r) : error("Estimated r=0. Noisy signal.")
end
