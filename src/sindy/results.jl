abstract type AbstractSparseIdentificationResult end;

mutable struct SparseIdentificationResult <: AbstractSparseIdentificationResult
    coeff
    parameters
    equations

    iterations
    error
    sparsity
end

Base.show(io::IO, x::SparseIdentificationResult) = print(io, "Sparse Identification Result with $(x.sparsity) active terms.")

@inline function Base.print(io::IO, x::SparseIdentificationResult)
    println("Sparse Identification Result")
    println("Active terms : $(x.sparsity)")
    println("Training error : $(x.error)")
    println("Iterations : $(x.iterations)")
end

function SparseIdentificationResult(coeff, parameters, equations, iters, Y, X)
    error = norm(Y-coeff'*equations(X, p = parameters), 2)
    sparsity = Int64(norm(coeff, 0))
    return SparseIdentificationResult(coeff, parameters, equations, iters, error, sparsity)
end
