"""
$(TYPEDEF)

Scales the losses in such a way that the minimum loss is equal to one.
"""
struct RelativeReward{risk} <: AbstractRewardScale{risk} end

RelativeReward(risk_seeking = true) = RelativeReward{risk_seeking}()

function (::RelativeReward)(losses::Vector{T}) where {T <: Number}
    return exp.(minimum(losses) .- losses)
end

function (::RelativeReward{true})(losses::Vector{T}) where {T <: Number}
    r = exp.(minimum(losses) .- losses)
    return r .- minimum(r)
end

"""
$(TYPEDEF)

Scales the losses in such a way that the minimum loss is the most influential reward.
"""
struct AbsoluteReward{risk} <: AbstractRewardScale{risk} end

AbsoluteReward(risk_seeking = true) = AbsoluteReward{risk_seeking}()

(::AbsoluteReward)(losses::Vector{T}) where {T <: Number} = exp.(-losses)

function (::AbsoluteReward{true})(losses::Vector{T}) where {T <: Number}
    r = exp.(-losses)
    return r .- minimum(r)
end
