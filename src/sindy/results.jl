abstract type AbstractSparseIdentificationResult end;

mutable struct SparseIdentificationResult <: AbstractSparseIdentificationResult
    coeff::AbstractArray
    parameters::AbstractArray
    equations::Union{Function, Basis}

    opt::Union{Optimize.AbstractOptimizer, Optimize.AbstractSubspaceOptimizer}
    iterations::Int64
    converged::Bool

    error::AbstractArray
    aicc::AbstractArray
    sparsity::AbstractArray
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


"""
    print_equations(res; show_parameter)

Print the equations stored inside the `SparseIdentificationResult` `res`. If `show_parameter` is set
to true, the numerical values will be used. Otherwise, the symbolic form will appear.
"""
function print_equations(r::SparseIdentificationResult; show_parameter::Bool = false)
    if show_parameter
        eqs = r.equations(variables(r.equations), parameters(r), independent_variable(r.equations))
        for (i, eq) in enumerate(eqs)
            println("f_$i = ", eq)
        end
    else
        print(r.equations)
    end
end

"""
    SparseIdentificationResult()

Contains the result of a sparse identification. Contains the coefficient matrix `Ξ`,
the equations of motion, and its associated parameters. It also stores the optimizer, iteration counter, and convergence status.

Additionally, the model is evaluated over the training data and the ``L_2``-error, Akaikes Information Criterion, and the ``L_0``-Norm of the coefficients
is stored.
"""
function SparseIdentificationResult(coeff::AbstractArray, equations::Basis, iters::Int64, opt::T , convergence::Bool, Y::AbstractVecOrMat, X::AbstractVecOrMat; p::AbstractArray = [], t::AbstractVector = []) where T <: Union{Optimize.AbstractOptimizer, Optimize.AbstractSubspaceOptimizer}
    Ŷ = coeff'*equations(X, p, t)
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
    b_ = Basis(Operation[], variables(b), parameters = [parameters(b)...; p...], iv = independent_variable(b))

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
        if !isnothing(eq)
            push!(b_, eq)
        end
    end
    b_, p_
end

Base.size(r::SparseIdentificationResult) = size(r.sparsity)
Base.length(r::SparseIdentificationResult) = length(r.sparsity)

ModelingToolkit.parameters(r::SparseIdentificationResult) = r.parameters

dynamics(b::SparseIdentificationResult) = dynamics(b.equations)

"""
    get_sparsity(res)

Return the ``L_0``-Norm of the `SparseIdentificationResult`
"""
get_sparsity(r::SparseIdentificationResult) = r.sparsity

"""
    get_error(res)

Return the ``L_2``-Error of the `SparseIdentificationResult` over the training data.
"""
get_error(r::SparseIdentificationResult) = r.error

"""
    get_aicc(res)

Return Akaikakes Information Criterion of the `SparseIdentificationResult` over the training data.
"""
get_aicc(r::SparseIdentificationResult) = r.aicc

"""
    get_coefficients(res)

Return the coefficient matrix `Ξ` of the `SparseIdentificationResult`.
"""
get_coefficients(r::SparseIdentificationResult) = r.coeff

function ModelingToolkit.ODESystem(b::SparseIdentificationResult)
    return ODESystem(b.equations)
end
