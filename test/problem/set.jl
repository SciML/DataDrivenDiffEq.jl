using Revise
using DataDrivenDiffEq
using ModelingToolkit
using Plots

X = randn(3, 10)
Y = randn(3, 20)
t1 = float.(collect(1:1:10))
t2 = float.(collect(1:1:20))

data = (prob1 = (X = X, t = t1), prob2 = (X = Y, t = t2))

problem_kwargs(nt::NamedTuple; kwargs...) = begin
    kys = Tuple([k for k in keys(nt) if k != :X])
    merge(nt[kys], kwargs)
end


dset = my_test(DataDrivenDiffEq.Continuous ,data)
DataDrivenDiffEq.get_oop_args(dset)

@parameters t
@variables x y 
b = Basis([x; y; t], [x;y], independent_variable = t)
plot(b(dset))
DX = zeros(3, length(dset))
b(DX, dset)
DX == b(dset)

DataDrivenDiffEq.get_implicit_oop_args(dset)

DataDrivenDiffEq.DirectDataset(data)
