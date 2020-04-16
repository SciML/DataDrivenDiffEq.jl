is_discrete(k::AbstractKoopmanOperator) = k.discrete
is_continouos(k::AbstractKoopmanOperator) = !k.discrete

LinearAlgebra.eigen(k::AbstractKoopmanOperator) = !isempty(k.operator) ? eigen(k.operator) : "No discrete operator available!"
LinearAlgebra.eigvals(k::AbstractKoopmanOperator) = !isempty(k.operator) ? eigvals(k.operator) : "No discrete operator available!"
LinearAlgebra.eigvecs(k::AbstractKoopmanOperator) = !isempty(k.operator) ? eigvecs(k.operator) : "No discrete operator available!"

modes(k::AbstractKoopmanOperator) = is_continouos(k) ? eigenvecs(k) : "Koopman is discrete."
frequencies(k::AbstractKoopmanOperator) = is_continouos(k) ? eigvals(k) : "Koopman is discrete."

operator(k::AbstractKoopmanOperator) = is_discrete(k) ? k.operator : "Koopman is continouos."
generator(k::AbstractKoopmanOperator) = is_continouos(k) ? k.operator : "Koopman is discrete."

inputmap(k::AbstractKoopmanOperator) = k.input
outputmap(k::AbstractKoopmanOperator) = k.output

updateable(k::AbstractKoopmanOperator) = !isempty(k.Q) && !isempty(k.P)
isstable(k::AbstractKoopmanOperator) = is_discrete(k) ? all(abs.(eigvals(k)) .< one(eltype(k.operator))) : all(real.(eigvals(k)) < zero(eltype(k.operator)))
