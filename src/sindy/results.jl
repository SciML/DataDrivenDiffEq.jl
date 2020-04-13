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

Base.show(io::IO, x::SparseIdentificationResult) = print(io, "Sparse Identification Result with $(x.sparsity) active terms.")

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
    println("AICC : $(x.aicc)\n")

    print("$(x.opt)")
    if x.converged
        println(" converged after $(x.iterations) iterations.")
    else
        println(" did not converge after $(x.iterations) iterations.")
    end
end

function SparseIdentificationResult(coeff::AbstractArray, equations::Basis, iters::Int64, opt::T , convergence::Bool, Y::AbstractVecOrMat, X::AbstractVecOrMat; p::AbstractArray = []) where T <: Union{Optimise.AbstractOptimiser, Optimise.AbstractSubspaceOptimiser}
    error = norm.(eachrow(Y-coeff'*equations(X)), 2)
    sparsity = Int64.(norm.(eachcol(coeff), 0))
    aicc = AICC(sum(sparsity), coeff'*equations(X), Y)
    b_, p_ = derive_parameterized_eqs(coeff, equations, sum(sparsity))
    return SparseIdentificationResult(coeff, [p...;p_...], b_ , opt, iters, convergence,  error, aicc,  sparsity)
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

ModelingToolkit.parameters(r::SparseIdentificationResult) = r.parameters

function ModelingToolkit.ODESystem(b::SparseIdentificationResult)
    return ODESystem(b.equations)
end

function ModelingToolkit.ODESystem(b::SparseIdentificationResult, independent_variable::Operation)
    return ODESystem(b.equations, independent_variable)
end

dynamics(b::SparseIdentificationResult) = dynamics(b.equations)
