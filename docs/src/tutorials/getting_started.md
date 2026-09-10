*End-to-end workflow for solving the Capacitated Facility Location Problem (CFLP) using built-in BendersX.jl components*

## 1. Mathematical Modeling
### [Parameters](@id cflp-parameter)
- number of facilities ``I``
- number of customers ``J``
- facility capacities ``\mathbf{b} = (b_i)_{i \in [I]}``
- customer demands ``\mathbf{d} = (d_j)_{j \in [J]}``
- fixed opening cost ``\mathbf{f} = (f_i)_{i \in [I]}``
- transportation cost ``\mathbf{c} = (c_{ij})_{i \in [I], j \in [J]}``. 

The CFLP can be written as the following mixed-integer program:
```math 
    \begin{align}
    \min \ & \sum_{i \in [I]}f_ix_i + \sum_{i \in [I], j \in [J]}c_{ij}d_j y_{ij}\nonumber \\
    \text{s.t.} \ & \sum_{i \in [I]} y_{ij } = 1, \ \forall j \in [J],\nonumber \\
    & y_{ij} \le x_i, \ \forall i \in [I], j \in [J],\nonumber \\
    & \sum_{j \in [J]}d_j y_{ij} \le b_i x_i, \ \forall i \in [I]\nonumber \\
    & y_{ij} \ge 0, \ \forall i \in [I], j \in [J],\nonumber \\
    &x_i \in \mathbb B, \ \forall i \in [I],\nonumber 
\end{align}
```
- The first constraint enforces demand fulfillment.
- The second links assignment decisions to facility-opening decisions for better LP relaxation.
- The third enforces facility capacity limits.

### Benders Reformulation 
Separating the binary variables and introducing an auxiliary variable ``t``, we obtain
#### [Master problem](@id cflp-master)
```math 
    \begin{align}
    \min \ & \mathbf{f}^\top \mathbf{x} + t \nonumber \\
    \text{s.t.} \ 
    &x_i \in \mathbb B, \ \forall i \in [I]. \nonumber 
    \end{align}
```
Two redundant constraints may optionally be added:
```math 
\begin{align}
     & \mathbf{b}^\top \mathbf{x} \ge \mathbf{1}^\top \mathbf{d}, \nonumber \\
    & t \ge -10^{6}, \nonumber
\end{align}
```
where the first ensures sufficient total capacity and the second prevents 
``t`` from initially taking an arbitrarily negative value.

#### [Subproblem](@id cflp-sub)
```math
\begin{align}
    \min \ & \sum_{i \in [I], j \in [J]}c_{ij}d_j y_{ij} \nonumber \\
    \text{s.t.} \ & \sum_{i \in [I]} y_{ij } = 1, \ \forall j \in [J],\nonumber \\
    & y_{ij} \le x_i, \ \forall i \in [I], j \in [J],\nonumber \\
    & \sum_{j \in [J]}d_j y_{ij} \le b_i x_i, \ \forall i \in [I]\nonumber \\
    & y_{ij} \ge 0, \ \forall i \in [I], j \in [J].\nonumber \\
\end{align}
```




## 2. Full Code Example
The following code block illustrates how to solve a CFLP instance using a sequential Benders decomposition with classical optimality and feasibility cuts.

```julia
using BendersX
using JuMP, CPLEX

struct CFLPData
    n_facilities::Int
    n_customers::Int
    capacities::Vector{Float64}
    demands::Vector{Float64}
    fixed_costs::Vector{Float64}
    costs::Matrix{Float64}
end

function read_cflp_benchmark_data(path_to_raw_data)
    """
    Users are responsible for loading their raw data into the 
    user-defined data structure (e.g., `CFLPData`). See, for example,
    `read_cflp_benchmark_data` in
    `src/problems/cflp/data_reader.jl`.
    """
end

function update_master_model!(model::Model, data::CFLPData)
    I = data.n_facilities
    @variable(model, x[1:I], Bin)
    @variable(model, t >= -1e6)
    @objective(model, Min, data.fixed_costs'* x + t)
    @constraint(model, capacity, sum(data.capacities[i] * x[i] for i in 1:I) >= sum(data.demands))

    return (x = x, ), t
end

function update_sub_model!(model::Model, data::CFLPData, scen_idx::Int; x)
    I, J = data.n_facilities, data.n_customers   
    @variable(model, y[1:I, 1:J] >= 0)
    cost_demands = data.costs .* data.demands'
    @objective(model, Min, sum(cost_demands .* y))
    @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
    @constraint(model, facility_open, y .<= x)
    @constraint(model, capacity[i in 1:I], sum(data.demands[:] .* y[i,:]) <= data.capacities[i] * x[i])
end

data   = read_cflp_benchmark_data("p1")
master_optimizer = optimizer_with_attributes(CPLEX.Optimizer, MOI.Silent() => true)
subproblem_optimizer = optimizer_with_attributes(
    CPLEX.Optimizer,
    "CPXPARAM_Threads" => 7,
    MOI.Silent() => true,
)
master = Master(data; model = update_master_model!, optimizer = master_optimizer)
oracle = ClassicalOracle(data, master; model = update_sub_model!, optimizer = subproblem_optimizer)
env    = BendersSeq(master, oracle)
log    = solve!(env)
```

Attach solvers through standard JuMP APIs such as `optimizer_with_attributes(...)`.




## 2. Data 
*Providing Instance Data to the Benders Engine*

Problem data can be stored in any Julia object that provides the parameters needed to construct the master and subproblem models.

### Example
The [parameters](@ref cflp-parameter) of the Capacitated Facility Location Problem (CFLP) may be written as:
```julia
struct CFLPData
    n_facilities::Int
    n_customers::Int
    capacities::Vector{Float64}
    demands::Vector{Float64}
    fixed_costs::Vector{Float64}
    costs::Matrix{Float64}
end
```
### Loading Data
Users are responsible for loading and preprocessing raw instance data into their chosen data container. A typical pattern is:
```julia
read_data(path_to_raw_data) -> MyData
```
For an example specific to CFLP, see `read_cflp_benchmark_data` in `src/problems/cflp/data_reader.jl`.

## [3. Modeling Interface](@id modeling-interface)
*Defining Master and Subproblem Models in BendersX.jl*

Users provide the master and subproblem formulations through model-update functions written in standard JuMP syntax.
These functions can have any name; the examples below use `update_*` names and pass them through the `model` keyword.
If you are unfamiliar with JuMP, please refer to the JuMP.jl documentation for an introduction:
[Julia JuMP](https://jump.dev/JuMP.jl/stable/)

### Master Modeling
Users specify the master formulation by implementing a function of the form:
```julia
update_master_model!(model::Model, data) -> NamedTuple, Vector{VariableRef}
```
Within this function, users use standard JuMP commands to declare master-level variables, constraints, and the objective.
The function must return:
1. a NamedTuple mapping symbolic variable names to non-auxiliary master variables, and
2. a Vector{VariableRef} containing the auxiliary variables $t$ used for Benders cuts.

All modeling decisions are expressed in the model-update functions, while solver attachment remains explicit through constructor keywords such as `optimizer`.
The Benders engine itself remains independent of the model and the solver.

### Example
[The CFLP master problem](@ref cflp-master) can be implemented as:

```julia
function update_master_model!(model::Model, data::CFLPData)
    I = data.n_facilities
    @variable(model, x[1:I], Bin)
    @variable(model, t >= -1e6)
    @objective(model, Min, data.fixed_costs'* x + t)
    @constraint(model, capacity, sum(data.capacities[i] * x[i] for i in 1:I) >= sum(data.demands))

    return (x = x, ), t
end
```

Then attach the optimizer when constructing the master:

```julia
master = Master(
    data;
    model = update_master_model!,
    optimizer = optimizer_with_attributes(CPLEX.Optimizer, MOI.Silent() => true),
)
```

### Subproblem Modeling
Subproblems are specified by the user through a model-update function:
```julia
update_sub_model!(model::Model, data, scen_idx::Int; kwargs...)
```
Here, `kwargs...` contains the symbolic names of the master variables that appear in the subproblem. This allows users to formulate the subproblem in JuMP **while referencing these master variables directly**, without explicitly adding them to the subproblem model.

### Example 1
[The CFLP subproblem](@ref cflp-sub) can be implemented like this:
```julia
function update_sub_model!(model::Model, data::CFLPData, scen_idx::Int; x)
    I, J = data.n_facilities, data.n_customers   
    @variable(model, y[1:I, 1:J] >= 0)
    cost_demands = data.costs .* data.demands'
    @objective(model, Min, sum(cost_demands .* y))
    @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
    @constraint(model, facility_open, y .<= x)
    @constraint(model, capacity[i in 1:I], sum(data.demands[:] .* y[i,:]) <= data.capacities[i] * x[i])
end
```

Attach the subproblem optimizer when constructing the oracle:

```julia
oracle = ClassicalOracle(
    data,
    master;
    model = update_sub_model!,
    optimizer = optimizer_with_attributes(
        CPLEX.Optimizer,
        "CPXPARAM_Threads" => 7,
        MOI.Silent() => true,
    ),
)
```

### Example 2
Consider the following master and subproblem formulations:
#### Master problem
```math 
    \begin{align}
    \min \ & t \nonumber \\
    \text{s.t.} \ 
    & \sum_{i \in [10]} u_i \ge 2,\nonumber \\
    &u_i \in \mathbb B, \ \forall i \in [10], \nonumber\\
    &v_{i,j} \in \mathbb B, \ \forall i = 3, \cdots, 10, j \in \{A, B\}, \nonumber \\ 
    &w_{i,j} \in \mathbb B, \ \forall i = 1, \cdots, 3, j = i, \cdots, 10, \nonumber \\ 
    & t \ge -10^{6} \nonumber 
    \end{align}
```
#### Subproblem
```math 
    \begin{align}
    \min \ & \sum_{i \in [10]} y_i \nonumber \\
    \text{s.t.} \ 
    & y_i \le u_i, \ \forall i \in [10],\nonumber \\
    & y_1 = v_{10,A},\nonumber \\
    & y_2 = w_{3,10},\nonumber \\
    &y_i \ge 0, \ \forall i \in [10]. \nonumber
    \end{align}
```

These problems can be implemented as follows:
```julia
function update_master_model!(model::Model, data::EmptyData)
    @variable(model, u[1:10], Bin) # Array
    @variable(model, v[3:10, [:A, :B]], Bin) # DenseAxisArray
    @variable(model, w[i=1:3,j=i:10], Bin) # SparseAxisArray
    @variable(model, t >= -1e6)
    @constraint(model, sum(u) >= 2)
    @objective(model, Min, 1.0 * t)
    
    return (u = u, v = v, w = w), t
end

function update_sub_model!(model::Model, data::EmptyData, scen_idx::Int; u, v, w)
    @variable(model, y[1:10] >= 0)
    @objective(model, Min, sum(y))
    @constraint(model, y .<= u)
    @constraint(model, y[1] = v[10,:A])
    @constraint(model, y[2] = w[3,10])
end
```

!!! tip "Common Pitfalls"

    When defining a subproblem model-update function, keep the following points in mind:

    * **Explicit keyword names**: The keyword argument names in the subproblem function must exactly match the names returned by the master model-update function.
    * **No redeclaration**: Do not redeclare master variables inside the subproblem; they should only be referenced via keyword arguments.
    * **Indexing with symbolic sets**: When using `DenseAxisArray` or `SparseAxisArray`, ensure that symbolic indices (e.g., `:A`) are used consistently.
    * **Scenario index usage**: If `scen_idx` is unused, it can be safely ignored, but it must still appear in the function signature.

---

## 4. Running Benders

Users can run Benders decomposition by selecting appropriate combinations of an **oracle** and an **environment**. A summary of the built-in Oracle and Environment variants is provided in the *Plug-and-Play* tip in the [Quick Start](@ref quick-start).

Both oracles and environments can be further customized using dedicated parameter objects or by composing alternative sub-components, allowing fine-grained control over algorithmic behavior.

!!! info "Next steps"
    - Refer to **Tutorials / Swapping Oracles and Adjusting Their Behaviors** for oracle configuration and customization.
    - Refer to **Tutorials / Swapping Environments and Adjusting Their Behaviors** for execution logic customization.
    - Refer to **Tutorials / Examples** for a collection of worked examples.
