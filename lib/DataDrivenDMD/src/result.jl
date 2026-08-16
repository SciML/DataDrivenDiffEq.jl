"""
$(TYPEDEF)

Result returned by DataDrivenDMD solvers.

# Fields

$(FIELDS)

The `k`, `b`, and `c` fields represent the learned operator, input map, and
output map. `q` and `p` retain update matrices used by the online formulation;
they are developer state and should not be edited by callers. The remaining
fields implement the `StatsAPI.StatisticalModel` interface.

# Returns

The constructor returns a result whose operator and maps are compatible with
`get_operator`, `get_inputmap`, and `get_outputmap`.
"""
struct KoopmanResult{K, B, C, Q, P, T} <: AbstractDataDrivenResult
    """Matrix representation of the operator / generator"""
    k::K
    """Matrix representation of the inputs mapping"""
    b::B
    """Matrix representation of the pullback onto the states"""
    c::C
    """Internal matrix used for updating"""
    q::Q
    """Internal matrix used for updating"""
    p::P
    # Statistical-model results
    """Residual sum of squares"""
    rss::T
    """Loglikelihood"""
    loglikelihood::T
    """Nullloglikelihood"""
    nullloglikelihood::T
    """Degrees of freedom"""
    dof::Int
    """Number of observations"""
    nobs::Int

    """Returncode"""
    retcode::DDReturnCode

    function KoopmanResult(
            k_::K, b::B, c::C, q::Q, p::P, X::AbstractMatrix{T},
            Y::AbstractMatrix{T}, U::AbstractMatrix
        ) where {K, B, C, Q, P, T}
        k = Matrix(k_)
        rss = isempty(b) ? sum(abs2, Y .- c * k * X) : sum(abs2, Y .- c * (k * X .+ b * U))
        dof = sum(!iszero, k)
        dof += isempty(b) ? 0 : sum(!iszero, b)
        nobs = prod(size(Y))
        ll = -nobs / 2 * log(rss / nobs)
        nll = -nobs / 2 * log(mean(abs2, Y .- vec(mean(Y, dims = 2))))

        return new{K, B, C, Q, P, T}(k_, b, c, q, p, rss, ll, nll, dof, nobs, DDReturnCode(1))
    end
end

is_success(k::KoopmanResult) = getfield(k, :retcode) == DDReturnCode(1)

"""
$(SIGNATURES)

Return the learned Koopman operator or generator matrix.
"""
get_operator(k::KoopmanResult) = getfield(k, :k)
get_generator(k::KoopmanResult) = getfield(k, :k)

"""
$(SIGNATURES)

Return the learned input map.
"""
get_inputmap(k::KoopmanResult) = getfield(k, :b)

"""
$(SIGNATURES)

Return the learned output map.
"""
get_outputmap(k::KoopmanResult) = getfield(k, :c)

# StatsAPI interface
StatsAPI.coef(x::KoopmanResult) = getfield(x, :k)

StatsAPI.rss(x::KoopmanResult) = getfield(x, :rss)

StatsAPI.dof(x::KoopmanResult) = getfield(x, :dof)

StatsAPI.nobs(x::KoopmanResult) = getfield(x, :nobs)

StatsAPI.loglikelihood(x::KoopmanResult) = getfield(x, :loglikelihood)

StatsAPI.nullloglikelihood(x::KoopmanResult) = getfield(x, :nullloglikelihood)

StatsAPI.r2(x::KoopmanResult) = r2(x, :CoxSnell)
