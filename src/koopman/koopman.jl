is_discrete(k::AbstractKoopmanOperator) = k.discrete
is_continouos(k::AbstractKoopmanOperator) = !k.discrete

LinearAlgebra.eigen(k::AbstractKoopmanOperator) = eigen(k.operator)
LinearAlgebra.eigvals(k::AbstractKoopmanOperator) = eigvals(k.operator)
LinearAlgebra.eigvecs(k::AbstractKoopmanOperator) = eigvecs(k.operator)

modes(k::AbstractKoopmanOperator) = is_continouos(k) ? eigvecs(k) : throw(AssertionError("Koopman is discrete."))
frequencies(k::AbstractKoopmanOperator) = is_continouos(k) ? eigvals(k) : throw(AssertionError("Koopman is discrete."))

operator(k::AbstractKoopmanOperator) = is_discrete(k) ? k.operator : throw(AssertionError("Koopman is continouos."))
generator(k::AbstractKoopmanOperator) = is_continouos(k) ? k.operator : throw(AssertionError("Koopman is discrete."))

inputmap(k::AbstractKoopmanOperator) = k.input
outputmap(k::AbstractKoopmanOperator) = k.output

updateable(k::AbstractKoopmanOperator) = !isempty(k.Q) && !isempty(k.P)
isstable(k::AbstractKoopmanOperator) = is_discrete(k) ? all(abs.(eigvals(k)) .< one(eltype(k.operator))) : all(real.(eigvals(k)) < zero(eltype(k.operator)))
