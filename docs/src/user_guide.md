# [User Guide](@id user-guide)

## BendersX Components
**BendersX.jl** is built on the principle that Benders decomposition consists of **clearly defined, separable components**, each with a distinct role. By making these components explicit, the framework enables **composable and extensible algorithm design**.

BendersX.jl decomposes the Benders decomposition algorithm into three core components: the **Master**, the **Oracle**, and the **Environment**.
Each component has a well-defined responsibility and can be independently replaced or extended.

### Master
The **Master** maintains the current master problem. It stores the optimization
model, incorporates Benders cuts, and provides the linking and auxiliary
variables used by the rest of the algorithm.

Conceptually, the Master:
- represents the current relaxation of the Benders reformulation
- refines this relaxation by incorporating newly generated Benders cuts, and
- produces candidate solutions for evaluation.

In BendersX.jl, the Master is:
- is encapsulated by a subtype of `AbstractMaster`
- owns a JuMP model defining the master problem, and
- accepts newly generated Benders cuts during the solution process.

There are two ways to customize the Master:

- **Change the master formulation.** Define a new `update_master_model!` method
  while retaining the provided `Master` implementation.
- **Change the master implementation.** Define a new subtype of
  `AbstractMaster` and implement its public interface.

These two mechanisms are independent. A new master formulation does not
require a new `AbstractMaster` implementation.

### Adding a New Master

For most problems, users only need to define the master formulation through
`update_master_model!` and use the provided `Master` implementation.

A new `AbstractMaster` subtype is needed only when the master representation
or master-side behavior itself should change. Custom implementations interact
with the rest of BendersX through a small public interface:

- `master_model` provides the underlying JuMP model;
- `linking_variables` and `auxiliary_variables` provide the variables used by
  the Benders algorithm;
- `copy_linking_variable_tuple!` provides the structured linking variables
  used to construct subproblem models;
- `evaluate_objective` evaluates the original objective at supplied linking
  and auxiliary values; and
- `add_cuts!` incorporates generated Benders cuts into the master.

A custom Master is therefore free to use its own internal representation. For
example:

```julia
struct MyMaster <: AbstractMaster
    jump_model::JuMP.Model
    linking_vars::Vector{JuMP.VariableRef}
    auxiliary_vars::Vector{JuMP.VariableRef}
    # additional fields
end

BendersX.master_model(master::MyMaster) = master.jump_model
BendersX.linking_variables(master::MyMaster) = master.linking_vars
BendersX.auxiliary_variables(master::MyMaster) = master.auxiliary_vars

# Implement the remaining AbstractMaster interface as required.

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
