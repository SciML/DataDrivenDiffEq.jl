# Result selection
select_by(x, y::AbstractMatrix; kwargs...) = y 
select_by(x, sol; kwargs...) = select_by(Val(x), sol)

## Koopman

select_by(::Val, sol::AbstractKoopmanSolution; kwargs...) = begin
    @unpack k, error = sol
    i = argmin(error)
    return k[i], error[i]
end

select_by(::Val{:kfold}, sol::AbstractKoopmanSolution; weights = false, kwargs...) = begin
    @unpack k, folds, error  = sol
    size(k, 1) <= 1 && return select_by(1, sol)
    i = argmin(mean(folds, dims = 1)[1,:])
    return k[i], error[i]
end

## Sparse Regression
select_by(::Val, sol::AbstractSparseSolution; kwargs...) = begin
    @unpack Ξ, error, λ = sol
    i = argmin(error)
    return Ξ[i,:,:], error[i], λ[i,:]
end

select_by(::Val{:kfold}, sol::AbstractSparseSolution; weights = false, kwargs...) = begin
    @unpack Ξ, folds, error, λ = sol
    size(Ξ, 1) <= 1 && return select_by(1, sol)
    i = argmin(mean(folds, dims = 1)[1,:])
    return Ξ[i,:,:], error[i], λ[i,:]
end

select_by(::Val{:stat}, sol::AbstractSparseSolution; weights = false, kwargs...) = begin
    @unpack Ξ, folds, error, λ = sol
    size(Ξ, 1) <= 1 && return select_by(1, sol)
    best = argmin(error)
    ξ = mean(Ξ, dims = 1)[1,:,:]
    s = std(Ξ, dims = 1)[1,:,:]
    return measurement.(ξ, s), error[best], λ[best,:]
end

## Ensembles
select_by(by::Val, sol::AbstractVector{T}; kwargs...) where T <: AbstractKoopmanSolution = begin
    results_ = map(sol) do r
        select_by(by, r)
    end

    xis = cat(map(x->reshape(x, 1, size(x)...), first.(results_))..., dims = 1)
    errors = map(x->x[2], results_)
    # Weights
    w = Weights(1 .- (errors .- minimum(errors)) ./ (maximum(errors) - minimum(errors)))
    
    # Take the average of the threshold
    Ξ = mean(xis, w, dims = 1)
    Ξ_std =  reshape(std(xis, w, 1, mean = Ξ), size(Ξ, 2), size(Ξ, 3))
    Ξ = reshape(Ξ, size(Ξ, 2), size(Ξ, 3))

    measurement.(Ξ, Ξ_std), mean(errors, w), vec(w)
end

select_by(by::Val, sol::AbstractVector{T}; kwargs...) where T <: AbstractSparseSolution= begin
    results_ = map(sol) do r
        select_by(by, r)
    end

    xis = cat(map(x->reshape(x, 1, size(x)...), first.(results_))..., dims = 1)
    errors = map(x->x[2], results_)
    lambdas = reduce(hcat, map(x->x[3], results_))
    # Weights
    w = Weights(1 .- (errors .- minimum(errors)) ./ (maximum(errors) - minimum(errors)))
    
    # Take the average of the threshold
    λ̄ = mean(lambdas, dims = 2)[:,1]
    Ξ = mean(xis, w, dims = 1)
    Ξ_std =  reshape(std(xis, dims = 1, mean = Ξ), size(Ξ, 2), size(Ξ, 3))
    Ξ = reshape(Ξ, size(Ξ, 2), size(Ξ, 3))

    for i in size(Ξ, 2)
        idxs = abs.(Ξ[:, i]) .< λ̄[i]
        Ξ[idxs, i] .= zero(eltype(Ξ))
        Ξ_std[idxs, i] .= zero(eltype(Ξ))
    end

    measurement.(Ξ, Ξ_std), mean(errors, w), λ̄, vec(w)
end
