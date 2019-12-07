function ExactDMD(X::AbstractArray; dt::T = 0.0, method::Symbol = :PINV, kwargs...) where T <: Real
    return ExactDMD(X[:, 1:end-1], X[:, 2:end], dt , method , kwargs)
end

function ExactDMD(X::AbstractArray, Y::AbstractArray, dt::T, method::Symbol, kwargs...) where T <: Real
    if method == :PINV
        return koopman_pinv(X, Y, :ExactDMD, dt)
    elseif method == :SVD

        return koopman_svd(X, Y, :ExactDMD,  dt , size(X)[1] , 0.0)
    end
end
