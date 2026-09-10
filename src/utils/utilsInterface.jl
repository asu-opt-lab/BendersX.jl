"""
    update_master_model!(model::Model, data) -> (x, t)

Formulate the master problem for `data` in `model`.

When `Master(data)` is called, BendersX searches for an `update_master_model!` method defined for the type of `data`. This fallback method is called only if no such method exists.

To use `Master(data)` with a custom data type, define

    update_master_model!(model::Model, data::MyDataType) -> (x, t)

where `MyDataType` is the type that represents your problem data. BendersX does
not require it to have a particular supertype.

The function must:
1. Add the master variables, constraints, and objective to `model`.
2. Return `(x, t)`.

`x` must be a `NamedTuple` containing the master variables that appear in subproblems. 
The field names of `x` determine the keyword arguments passed to
[`update_sub_model!`](@ref).

`t` must be a `VariableRef` or `Vector{VariableRef}` representing the auxiliary
variable(s) used to approximate the subproblem objective value.

Optimizer selection is handled elsewhere. Do not attach an optimizer to `model` in this
function.

# Example

```julia
struct MyDataType
    n::Int
end

function update_master_model!(model::Model, data::MyDataType)
    @variable(model, u[1:data.n])
    @variable(model, v[1:data.n])
    @variable(model, theta)

    @constraint(model, sum(u) >= 1)
    @objective(model, Min, theta)

    return (u = u, v = v), theta
end
```

You may also pass a custom builder directly:

```julia
Master(data; model = build_master_model!)
```

In that case, `build_master_model!(model, data)` must follow the same interface.
"""
function update_master_model!(model::Model, data)
    throw(UnimplementedInterfaceException(
        "BendersX does not know how to formulate a master problem for " * "$(typeof(data)). Define " * "`update_master_model!(model::Model, data::$(typeof(data)))`, " * "or pass a custom model-building function with " * "`Master(data; model = your_builder!)`."
))
end

"""
    update_sub_model!(model::Model, data, scen_idx::Int; kwargs...)

Formulate the subproblem for scenario `scen_idx` using `data` in `model`.

When a model-based oracle is constructed, BendersX searches for an `update_sub_model!` method defined for the type of `data`. This fallback method is called only if no such method exists.

To use a model-based oracle with a custom data type, 

    update_sub_model!(model::Model, data::MyDataType, scen_idx::Int; kwargs...)

where `MyDataType` is the type that represents your problem data. BendersX does
not require it to have a particular supertype.

The method must:

1. Add the subproblem variables, constraints, and objective to `model`.
2. Use the master variables passed as keyword arguments, if needed.

The keyword arguments correspond to the fields of the `NamedTuple` returned by
[`update_master_model!`](@ref). For example, if the master-model builder returns

```julia
(x = x, y = y)
```

then the subproblem builder may be written as
```julia
function update_sub_model!(model, data, scen_idx; x, y)
    ...
end
```

The argument `scen_idx` identifies the scenario being formulated and may be
ignored for deterministic models.

Most implementations should return `nothing`. To define generalized bound
constraints (GBCs), return `(gbc_lhs, gbc_rhs, gbc_sense)`, where the three objects describe GBCs relating subproblem variables (`gbc_lhs`) to affine
expressions of master variables (`gbc_rhs`), with senses <=, >=, or ==. See
[`ClassicalOracle`](@ref) for an oracle that supports GBCs.

Optimizer selection is handled elsewhere. Do not attach an optimizer to
`model` in this function.

# Example

```julia
struct MyDataType
    demand::Vector{Float64}
    cost::Vector{Float64}
end

function update_sub_model!(model::Model, data::MyDataType, scen_idx::Int; x)
    @variable(model, y[eachindex(data.demand)] >= 0)

    @constraint(model, demand[j in eachindex(data.demand)], y[j] == data.demand[j])
    @constraint(model, linking[j in eachindex(data.demand)], y[j] <= x[j])
    @objective(model, Min, sum(data.cost[j] * y[j] for j in eachindex(data.demand)))

    return nothing
end
```

You may pass a custom builder directly:

```julia
ClassicalOracle(data, master; model = build_sub_model!)
```

In that case, `build_sub_model!(model, data, scen_idx; kwargs...)` must follow
the same interface.
"""
function update_sub_model!(model::Model, data, scen_idx::Int; kwargs...)
    throw(UnimplementedInterfaceException( 
        "BendersX does not know how to formulate a subproblem for " * "$(typeof(data)). Define " * "`update_sub_model!(model::Model, data::$(typeof(data)), " * "scen_idx::Int; kwargs...)`, or pass a custom model-building " *
        "function with `Oracle(...; model = your_builder!)`." )) 
end
