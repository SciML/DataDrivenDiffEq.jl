# Taken from https://royalsocietypublishing.org/doi/pdf/10.1098/rspa.2017.0009
function AIC(k::Int64, X::AbstractArray, Y::AbstractArray; likelyhood = (X,Y) -> sum(abs2, X-Y))
    return 2*k - 2*log(likelyhood(X, Y))
end
# Taken from https://royalsocietypublishing.org/doi/pdf/10.1098/rspa.2017.0009
function AICC(k::Int64, X::AbstractArray, Y::AbstractArray; likelyhood = (X,Y) -> sum(abs2, X-Y))
    @assert all(size(X) .== size(Y)) "Dimensions of trajectories should be equal !"
    return AIC(k, X, Y, likelyhood)+ 2*(k+1)*(k+2)/(size(X)[2]-k-2)
end

# Double check on that
# Taken from https://www.immagic.com/eLibrary/ARCHIVES/GENERAL/WIKIPEDI/W120607B.pdf
function BIC(k::Int64, X::AbstractArray, Y::AbstractArray; likelyhood = (X,Y) -> sum(abs2, X-Y))
    @assert all(size(X) .== size(Y)) "Dimensions of trajectories should be equal !"
    return - 2*log(likelyhood(X, Y)) + k*ln(size(X)[2])
end
