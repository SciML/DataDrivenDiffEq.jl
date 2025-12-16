# Local Difference operator for discrete-time systems
# This was removed from Symbolics.jl v7, so we define it locally for backwards compatibility
# See: https://github.com/SciML/DataDrivenDiffEq.jl/issues/563

using Symbolics: Operator, value, unwrap, wrap
using SymbolicUtils: term

"""
    Difference(t; dt, update=false)

Represents a difference operator for discrete-time systems.

# Fields

  - `t`: The independent variable
  - `dt`: The time step
  - `update`: If true, represents a shift/update operator

# Examples

```julia
@variables t
d = Difference(t; dt = 0.01)
```
"""
struct Difference <: Operator
    t
    dt
    update::Bool
    Difference(t; dt, update = false) = new(value(t), dt, update)
end

(D::Difference)(x) = term(D, unwrap(x))
(D::Difference)(x::Num) = wrap(D(unwrap(x)))

# More specific method to avoid ambiguity with SymbolicUtils.Operator method
SymbolicUtils.promote_symtype(::Difference, ::Type{T}) where {T} = T

function Base.show(io::IO, D::Difference)
    print(io, "Difference(", D.t, "; dt=", D.dt, ", update=", D.update, ")")
end
Base.nameof(::Difference) = :Difference

function Base.:(==)(D1::Difference, D2::Difference)
    isequal(D1.t, D2.t) && isequal(D1.dt, D2.dt) && isequal(D1.update, D2.update)
end
Base.hash(D::Difference, u::UInt) = hash(D.dt, hash(D.t, xor(u, 0x055640d6d952f101)))
