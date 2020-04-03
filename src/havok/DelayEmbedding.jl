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

# Optimal Hard Threshold Pruning for Singular Values of Henkel Matrix given q
function r_from_q(timeseries::AbstractArray, q::Int)
    H = HankelMatrix(timeseries, q)     # Henkel Matrix
    F = svd(H')                         # SVD fact
    return RankEstimation(F.S, size(H)) # RankEstimation
end


# Rank estimation controller of Gavish M., Donoho L., 2014 methods
function RankEstimation(S::Array{<:AbstractFloat,1}, size::Tuple{Int,Int})
    n, m = size
    β = m/n
    τ = optimal_SVD_HT(β,false) * median(S) * sqrt(m)
    r = length(findall(x->x.>τ, S))
    r > 0 ? (return r) : error("Estimated r=0. Noisy signal.")
end


#= function omega = optimal_SVHT_coef(beta, sigma_known)

 Coefficient determining optimal location of Hard Threshold for Matrix
 Denoising by Singular Values Hard Thresholding when noise level is known or
 unknown.

 See D. L. Donoho and M. Gavish, "The Optimal Hard Threshold for Singular
 Values is 4/sqrt(3)", http://arxiv.org/abs/1305.5870

 IN:
    beta: aspect ratio m/n of the matrix to be denoised, 0<beta<=1.
          beta may be a vector
    sigma_known: 1 if noise level known, 0 if unknown

 OUT:
    coef:   optimal location of hard threshold, up the median data singular
            value (sigma unknown) or up to sigma*sqrt(n) (sigma known);
            a vector of the same dimension as beta, where coef(i) is the
            coefficient correcponding to beta(i)

 Usage in known noise level:

   Given an m-by-n matrix Y known to be low rank and observed in white noise
   with mean zero and known variance sigma^2, form a denoised matrix Xhat by:

   [U D V] = svd(Y);
   y = diag(Y);
   y( y < (optimal_SVHT_coef(m/n,1) * sqrt(n) * sigma) ) = 0;
   Xhat = U * diag(y) * V';


 Usage in unknown noise level:

   Given an m-by-n matrix Y known to be low rank and observed in white
   noise with mean zero and unknown variance, form a denoised matrix
   Xhat by:

   [U D V] = svd(Y);
   y = diag(Y);
   y( y < (optimal_SVHT_coef_sigma_unknown(m/n,0) * median(y)) ) = 0;
   Xhat = U * diag(y) * V';

 -----------------------------------------------------------------------------
 Authors: Matan Gavish and David Donoho <lastname>@stanford.edu, 2013

 This program is free software: you can redistribute it and/or modify it under
 the terms of the GNU General Public License as published by the Free Software
 Foundation, either version 3 of the License, or (at your option) any later
 version.

 This program is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 details.

 You should have received a copy of the GNU General Public License along with
 this program.  If not, see <http://www.gnu.org/licenses/>.
 -----------------------------------------------------------------------------=#

function optimal_SVD_HT(β::Real, sigma_known::Bool)
    if sigma_known
        coef = optimal_SVD_HT_σ_known(β)
    else
        coef = optimal_SVD_HT_σ_unknown(β)
    end
    return coef
end

function optimal_SVD_HT_σ_known(β::Real)
    @assert all(β>0)
    @assert all(β<=1)

    w = (8 * β) ./ (β + 1 + sqrt(β.^2 + 14 * β +1))
    return sqrt.(2 * (β + 1) + w)
end

function optimal_SVD_HT_σ_unknown(β::Real)
    @assert all(β>0)
    @assert all(β<=1)

    coef = optimal_SVD_HT_σ_known(β)

    MPmedian = MedianMarcenkoPastur(β)

    return coef ./ sqrt.(MPmedian)
end

function MedianMarcenkoPastur(β::Real)
    MarPas = x -> 1-incMarPas(x,β,0)
    lobnd = (1 - sqrt(β))^2
    hibnd = (1 + sqrt(β))^2
    change = true

    while change && (hibnd - lobnd > .001)
        change = false
        x = collect(range(lobnd, length=5, stop=hibnd-100000000*eps()))
        y = zeros(length(x))
        for i in 1:length(x)
            y[i] = MarPas(x[i])
        end
        if any(y .< 0.5)
            lobnd = max(x[findall(x -> x < 0.5, y)]...)
            change = true
        end
        if any(y .> 0.5)
            hibnd = min(x[findall(x -> x > 0.5, y)]...)
            change = true
        end
    end
    return (hibnd+lobnd)./2
end

function incMarPas(x0::Real, β::Real, gamma::Real)
    if β > 1
        error("βBeyond")
    end
    topSpec = (1 + sqrt(β))^2
    botSpec = (1 - sqrt(β))^2
    MarPas = x -> IfElse((topSpec-x).*(x-botSpec) > 0,
                         sqrt((topSpec-x).*(x-botSpec))./(β.* x)./(2 .* pi),
                         0)
    if gamma != 0
       fun = x -> (x.^gamma .* MarPas(x))
    else
       fun = x -> MarPas(x)
    end
    I, err = quadgk(fun, x0, topSpec)
    return I
end

function IfElse(Q,point,counterPoint)
    y = point
    notQ = map(!,Q)
    if any(notQ)
        if length(counterPoint) == 1
            counterPoint = ones(size(Q)).*counterPoint
        end
        y(notQ) = counterPoint(notQ)
    end
    return y
end
