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
    # StatsBase results
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

    function KoopmanResult(k_::K, b::B, c::C, q::Q, p::P, X::AbstractMatrix{T},
            Y::AbstractMatrix{T}, U::AbstractMatrix) where {K, B, C, Q, P, T}
        k = Matrix(k_)
        rss = isempty(b) ? sum(abs2, Y .- c * k * X) : sum(abs2, Y .- c * (k * X .+ b * U))
        dof = sum(!iszero, k)
        dof += isempty(b) ? 0 : sum(!iszero, b)
        nobs = prod(size(Y))
        ll = -nobs / 2 * log(rss / nobs)
        nll = -nobs / 2 * log(mean(abs2, Y .- vec(mean(Y, dims = 2))))

        new{K, B, C, Q, P, T}(k_, b, c, q, p, rss, ll, nll, dof, nobs, DDReturnCode(1))
    end
end

is_success(k::KoopmanResult) = getfield(k, :retcode) == DDReturnCode(1)

get_operator(k::KoopmanResult) = getfield(k, :k)
get_generator(k::KoopmanResult) = getfield(k, :k)

get_inputmap(k::KoopmanResult) = getfield(k, :b)
get_outputmap(k::KoopmanResult) = getfield(k, :c)

# StatsBase Overload
StatsBase.coef(x::KoopmanResult) = getfield(x, :k)

StatsBase.rss(x::KoopmanResult) = getfield(x, :rss)

StatsBase.dof(x::KoopmanResult) = getfield(x, :dof)

StatsBase.nobs(x::KoopmanResult) = getfield(x, :nobs)

StatsBase.loglikelihood(x::KoopmanResult) = getfield(x, :loglikelihood)

StatsBase.nullloglikelihood(x::KoopmanResult) = getfield(x, :nullloglikelihood)

StatsBase.r2(x::KoopmanResult) = r2(x, :CoxSnell)
