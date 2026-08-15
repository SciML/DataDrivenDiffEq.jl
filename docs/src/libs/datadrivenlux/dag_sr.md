# DataDrivenLux

DataDrivenLux provides differentiable directed-acyclic-graph structure search
for discovering governing equations.

## Developer API

`AbstractDAGSRAlgorithm` is the extension interface for implementing another
search algorithm. Application code should use the concrete algorithms below.

```@docs
DataDrivenLux.AbstractDAGSRAlgorithm
DataDrivenLux.CommonAlgOptions
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
