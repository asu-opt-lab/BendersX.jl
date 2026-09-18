# [User Guide](@id user-guide)

## BendersX Components
**BendersX.jl** is built on the principle that Benders decomposition consists of **clearly defined, separable components**, each with a distinct role. By making these components explicit, the framework enables **composable and extensible algorithm design**.

BendersX.jl decomposes the Benders decomposition algorithm into three core components: the **Master**, the **Oracle**, and the **Environment**.
Each component has a well-defined responsibility and can be independently replaced or extended.

### Master
The **Master** represents the master problem in a Benders decomposition and is responsible for proposing candidate solutions.

Conceptually, the Master:
- represents the current relaxation of the Benders reformulation
- refines this relaxation by incorporating newly generated Benders cuts, and
- produces candidate solutions for evaluation.

In BendersX.jl, the Master is:
- is encapsulated by a subtype of `AbstractMaster`
- owns a JuMP model defining the master problem, and
- accepts newly generated Benders cuts during the solution process.

#### Two kinds of master customization

Supplying a new `update_master_model!` method changes the mathematical master formulation while retaining the package-provided `Master` implementation. This is the appropriate extension point for most new problem classes.

Defining a new subtype of `AbstractMaster` changes the implementation or master-side behavior itself. For example, a custom subtype may store its objects under different field names or override how generated cuts are added. These two extension mechanisms are independent.

### Adding a New Master

Built-in environments currently support JuMP-backed `AbstractMaster` implementations. A custom implementation provides the following public interface instead of reproducing the fields of `Master`:

```julia
using BendersX
using JuMP
using LinearAlgebra

struct MyMaster <: AbstractMaster
    jump_model::JuMP.Model
    linking_structure::NamedTuple
    linking_vars::Vector{JuMP.VariableRef}
    auxiliary_vars::Vector{JuMP.VariableRef}
    linking_costs::Vector{Float64}
    auxiliary_costs::Vector{Float64}
end

BendersX.master_model(master::MyMaster) = master.jump_model
BendersX.linking_variables(master::MyMaster) = master.linking_vars
BendersX.auxiliary_variables(master::MyMaster) = master.auxiliary_vars

BendersX.copy_linking_variables!(model::JuMP.Model, master::MyMaster) =
    BendersX.copy_variables!(model, master.linking_structure)

BendersX.evaluate_primal_objective(master::MyMaster, linking_vars, auxiliary_vars) =
    LinearAlgebra.dot(master.linking_costs, linking_vars) +
    LinearAlgebra.dot(master.auxiliary_costs, auxiliary_vars)
```

The order returned by `linking_variables` must match candidate linking-value vectors and the `a_x` coefficients of `Hyperplane`. Likewise, the order returned by `auxiliary_variables` must match candidate auxiliary-value vectors, oracle objective-value vectors, and `a_t`. Flattening the `NamedTuple` returned by
`copy_linking_variables!` must reproduce the same linking-variable order.

`add_cuts!` has a default implementation based on these methods. A custom master may override it to record or manage ordinary Benders cuts. This hook does not implement a cut-retention policy by itself.

Some components require additional capabilities from the underlying JuMP model:

- `SplitOracle` can transfer only the supported linear master constraints into its DCGLP.
- `LPRelaxationPreprocessing` requires master integrality constraints that JuMP can temporarily relax.
- `BendersBnB` requires an optimizer supporting the configured lazy-constraint and user-cut callbacks.

### Oracle
An **Oracle** encapsulates all procedures related to **cut generation** at a given separation point.

In BendersX.jl, an Oracle:
- is implemented as a subtype of `AbstractOracle`
- is fully decoupled from the Master and Environment, and
- focuses exclusively on cut generation.

Oracle behavior can be configured via parameters, such as violation tolerances and cut selection rules.

### Environment
The **Environment** controls the **execution logic** of the Benders algorithm.

While the Master and Oracle define *what* problems are solved and *how* cuts are generated, the Environment defines *how the algorithm proceeds*. Specifically, it:
- orchestrates the interaction between the Master and Oracle,
- manages iteration order and termination criteria, and
- handles logging, statistics, and output.

In BendersX.jl, an Environment:
- is implemented as a subtype of `AbstractBendersEnv`
- encapsulates the overall control flow of the algorithm, and
- enables alternative execution strategies to be expressed cleanly.

Environment behavior can be adjusted via parameters (e.g., stopping rules, stabilization dynamics) and configurable subcomponents (e.g., root preprocessing, callbacks).

--- 

## Hierarchical Architecture
Each component follows a clear type hierarchy that supports specialization and reuse. This hierarchy allows advanced users to extend existing implementations incrementally rather than implementing full components from scratch.

### Environment
![dd](EnvHierarchy.pdf)


### Oracle
![dd](OracleHierarchy.pdf)

## Adding New Oracles
Advanced users can implement custom cut generators by defining a new oracle:
```julia
struct MyOracle <: AbstractOracle
    # fields
end
```
together with the required interface:
```julia
function generate_cuts(
    oracle::MyOracle,
    x_value::Vector{Float64},
    t_value::Vector{Float64};
    tol_normalize = 1.0,
    time_limit = 3600,
)
    # Generate cuts that separate (x_value, t_value).
    return is_in_L, hyperplanes, sub_obj_vals
end
```
The returned tuple contains whether the candidate belongs to the relevant set,
the generated `Vector{Hyperplane}`, and the subproblem objective values used to
update the upper bound.
This interface-based design allows new algorithmic ideas to be prototyped directly within the framework.


## Adding New Environments
Custom execution logic can be implemented by defining a new Environment:
```julia
struct MyEnv <: AbstractBendersEnv
    # fields
end
```
together with the required execution method:
```julia
function solve!(env::MyEnv)::DataFrame
    # execution logic
end
```

---

## Rule of Thumb
- If you are changing *which cuts are generated*, customize the Oracle.
- If you are changing *how the algorithm runs*, customize the Environment.
