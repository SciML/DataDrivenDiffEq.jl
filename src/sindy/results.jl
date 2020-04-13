abstract type AbstractSparseIdentificationResult end;

mutable struct SparseIdentificationResult <: AbstractSparseIdentificationResult
    coeff::AbstractArray
    parameters::AbstractArray
    equations::Union{Function, Basis}

    opt
    iterations::Int64
    converged::Bool

    error
    aicc
    sparsity
end

Base.show(io::IO, x::SparseIdentificationResult) = print(io, "Sparse Identification Result with $(sum(x.sparsity)) active terms.")

@inline function Base.print(io::IO, x::SparseIdentificationResult)
    println("Sparse Identification Result")
    println("No. of Parameters : $(length(x.parameters))")
    println("Active terms : $(sum(x.sparsity))")
    for (i, si) in enumerate(x.sparsity)
        println("   Equation $i : $si")
    end

    println("Overall error (L2-Norm) : $(sum(x.error))")
    for (i, ei) in enumerate(x.error)
        println("   Equation $i : $ei")
    end
    println("AICC :")
    for (i, ai) in enumerate(x.aicc)
        println("   Equation $i : $ai")
    end

    print("\n$(x.opt)")
    if x.converged
        println(" converged after $(x.iterations) iterations.")
    else
        println(" did not converge after $(x.iterations) iterations.")
    end
end

function SparseIdentificationResult(coeff::AbstractArray, equations::Basis, iters::Int64, opt::T , convergence::Bool, Y::AbstractVecOrMat, X::AbstractVecOrMat; p::AbstractArray = []) where T <: Union{Optimise.AbstractOptimiser, Optimise.AbstractSubspaceOptimiser}
    Ŷ = coeff'*equations(X, p = p)
    training_error = norm.(eachrow(Y-Ŷ), 2)
    sparsity = Int64.(norm.(eachcol(coeff), 0))

    aicc = similar(training_error)
    for i in 1:length(aicc)
        aicc[i] = AICC(sparsity[i], view(Ŷ, i, :) , view(Y, i, :))
    end
    b_, p_ = derive_parameterized_eqs(coeff, equations, sum(sparsity))
    return SparseIdentificationResult(coeff, [p...;p_...], b_ , opt, iters, convergence,  training_error, aicc,  sparsity)
end


function derive_parameterized_eqs(Ξ::AbstractArray{T, 2}, b::Basis, sparsity::Int64) where T <: Real
    @parameters p[1:sparsity]
    p_ = zeros(eltype(Ξ), sparsity)
    cnt = 1
    b_ = Basis(Operation[], variables(b), parameters = [parameters(b)...; p...])

    for i=1:size(Ξ, 2)
        eq = nothing
        for j = 1:size(Ξ, 1)
            if !iszero(Ξ[j,i])
                if eq === nothing
                    eq = p[cnt]*b[j]
                else
                    eq += p[cnt]*b[j]
                end
                p_[cnt] = Ξ[j,i]
                cnt += 1
            end
        end
        push!(b_, eq)
    end
    b_, p_
end

Base.size(r::SparseIdentificationResult) = size(r.sparsity)
Base.length(r::SparseIdentificationResult) = length(r.sparsity)

ModelingToolkit.parameters(r::SparseIdentificationResult) = r.parameters

dynamics(b::SparseIdentificationResult) = dynamics(b.equations)

get_sparsity(r::SparseIdentificationResult) = r.sparsity
get_error(r::SparseIdentificationResult) = r.error
get_aicc(r::SparseIdentificationResult) = r.aicc
get_coefficients(r::SparseIdentificationResult) = r.coeff

function ModelingToolkit.ODESystem(b::SparseIdentificationResult)
    return ODESystem(b.equations)
end

function ModelingToolkit.ODESystem(b::SparseIdentificationResult, independent_variable::Operation)
    return ODESystem(b.equations, independent_variable)
end



(r::SparseIdentificationResult)(X::AbstractArray = []) = r.equations(X, p = r.parameters)
