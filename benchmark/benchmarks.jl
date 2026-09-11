using DataDrivenDiffEq, BenchmarkTools
using StableRNGs, LinearAlgebra
using Symbolics

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

# Synthetic data from a linear system
n_states = 5
n_t = 500
X = rand(rng, n_states, n_t)
DX = rand(rng, n_states, n_t)
t = collect(range(0.0, 10.0, length = n_t))
U = rand(rng, 2, n_t)

# =============================================================================
# Problem construction
# =============================================================================

SUITE["problem"] = BenchmarkGroup()

SUITE["problem"]["continuous_t"] = @benchmarkable ContinuousDataDrivenProblem(
    $X, $t
)
SUITE["problem"]["continuous_dx"] = @benchmarkable ContinuousDataDrivenProblem(
    $X, $DX
)
SUITE["problem"]["continuous_with_control"] = @benchmarkable ContinuousDataDrivenProblem(
    $X, $t, $U
)
SUITE["problem"]["discrete"] = @benchmarkable DiscreteDataDrivenProblem($X, $t)
SUITE["problem"]["direct"] = @benchmarkable DirectDataDrivenProblem($X, $DX)
SUITE["problem"]["kernel_interp"] = @benchmarkable ContinuousDataDrivenProblem(
    $X, $t, GaussianKernel()
)

# =============================================================================
# Basis construction
# =============================================================================

SUITE["basis"] = BenchmarkGroup()

@variables bvars[1:5]
SUITE["basis"]["polynomial_basis"] = @benchmarkable polynomial_basis(
    $(collect(bvars)), 3
)
SUITE["basis"]["monomial_basis"] = @benchmarkable monomial_basis(
    $(collect(bvars)), 3
)
SUITE["basis"]["fourier_basis"] = @benchmarkable fourier_basis(
    $(collect(bvars)), 2
)
SUITE["basis"]["Basis_construct"] = @benchmarkable Basis(
    $(sin.(bvars .+ 1)), $(collect(bvars))
)

# =============================================================================
# Data collation and shrinkage
# =============================================================================

SUITE["collocate"] = BenchmarkGroup()

SUITE["collocate"]["collocate_data"] = @benchmarkable collocate_data(
    $X, $t, EpanechnikovKernel()
)
SUITE["collocate"]["optimal_shrinkage"] = @benchmarkable optimal_shrinkage($X)
SUITE["collocate"]["optimal_shrinkage!"] = @benchmarkable optimal_shrinkage!(
    $(copy(X))
)

# =============================================================================
# Normalization (DataNormalization is a processing spec used in problem options)
# =============================================================================

SUITE["normalize"] = BenchmarkGroup()

SUITE["normalize"]["problem_with_normalize"] = @benchmarkable ContinuousDataDrivenProblem(
    $X, $t, GaussianKernel(); normalize = DataNormalization()
)
