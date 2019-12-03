function AIC(k::Int64, X::AbstractArray, Y::AbstractArray; likelyhood = (X,Y) -> sum(abs2, X-Y))
    @assert all(size(X) .== size(Y)) "Dimensions of trajectories should be equal !"
    return 2*k - 2*log(likelyhood(X, Y)) + 2*(k+1)*(k+2)/(size(X)[2]-k-2)
end
