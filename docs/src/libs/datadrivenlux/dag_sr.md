# DataDrivenLux

DataDrivenLux provides differentiable directed-acyclic-graph structure search
for discovering governing equations.

## Developer API

`AbstractDAGSRAlgorithm` is the extension interface for implementing another
search algorithm. Application code should use the concrete algorithms below.

```@docs
DataDrivenLux.AbstractDAGSRAlgorithm
DataDrivenLux.CommonAlgOptions
DataDrivenLux.init_model
DataDrivenLux.init_cache
DataDrivenLux.update_parameters!
DataDrivenLux.convert_to_basis
```

## Error Models

```@docs
AdditiveError
MultiplicativeError
ObservedModel
```

## Priors

```@docs
Softmax
GumbelSoftmax
DirectSimplex
```

## Search State

```@docs
DataDrivenLux.Dataset
DataDrivenLux.Candidate
DataDrivenLux.PathState
DataDrivenLux.FunctionNode
DataDrivenLux.FunctionLayer
DataDrivenLux.LayeredDAG
DataDrivenLux.SearchCache
```

## Rewards

```@docs
RelativeReward
AbsoluteReward
```

## Algorithms

```@docs
RandomSearch
Reinforce
CrossEntropy
```

## Developer API

The following interfaces are for implementing DataDrivenLux algorithms and
custom model components. Application code should use the concrete types above.

```@docs
DataDrivenLux.AbstractAlgorithmCache
DataDrivenLux.AbstractSimplex
DataDrivenLux.AbstractErrorModel
DataDrivenLux.AbstractRewardScale
```
