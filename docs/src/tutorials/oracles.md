```@meta
CurrentModule = BendersX
```

BendersX is designed to make **oracle selection and configuration modular**.
Users can swap oracle implementations and adjust their behavior without
changing the master problem or the execution environment.

This section explains:

1. How to replace one oracle with another
2. How oracle parameters affect cut generation
3. How disjunctive (`SplitOracle` with `LpDistanceNormalization`) oracles fit into the workflow

---

## Swapping Oracles

All Benders oracles in BendersX conform to the [`AbstractOracle`](@ref)
interface and are therefore interchangeable, provided they are compatible with
the underlying problem formulation (some oracles are problem-specific).

For example, switching from a classical Benders oracle to a knapsack-based
oracle for capacitated facility location problems requires only changing the oracle constructor:

```julia
# Classical Benders oracle
oracle = ClassicalOracle(data, master; model = update_sub_model!)

# Knapsack-based oracle (e.g., CFL)
oracle = CFLKnapsackOracle(data, master; model = update_sub_model!)
```

The execution environment (`BendersSeq`, `BendersSeqInOut`, `BendersBnB`, or any variants)
remains unchanged.

---

## Adjusting Oracle Parameters

Each oracle in BendersX owns a `param` field of type `<: AbstractOracleParam`,
which controls numerical tolerances and cut-generation behavior. By convention,
an oracle named `XOracle` uses a parameter type named `XOracleParam`.
See [`API`](@ref api) for detailed descriptions of oracle-specific parameters.

Common parameters include:

* `rtol`: relative tolerance for cut violation
* `atol`: absolute tolerance for cut violation
* `zero_tol`: numerical tolerance for detecting zero values.

### Example: ClassicalOracle

```julia
param = ClassicalOracleParam(rtol = 1e-6, atol = 1e-8)
oracle = ClassicalOracle(data, master; model = update_sub_model!, param = param)
```

Some oracles expose **behavioral parameters** in addition to numerical tolerances.

### Example: UFLKnapsackOracle

```julia
param = UFLKnapsackOracleParam(
            slim = true,
            add_only_violated_cuts = true,
            rtol = 1e-8,
        )
oracle = UFLKnapsackOracle(data; param = param)
```
Key behavioral options:
* `slim`: aggregate multiple cuts into a single hyperplane
* `add_only_violated_cuts`: discard non-violated cuts

These options can have a significant impact on performance and memory usage.

---

## Oracles in Separable or Multi-Scenario Settings
When the subproblem is separable (for example, in a multi-scenario setting),
users can employ [`SeparableOracle`](@ref) to manage multiple independent
subproblems.

With `SeparableOracle`, the subproblem oracle is specified as a **type
parameter**, making it straightforward to swap different oracle
implementations:

```julia
oracle = SeparableOracle(
    data,
    master,
    ClassicalOracle,
    N;
    sub_oracle_param = ClassicalOracleParam(rtol = 1e-6),
)
```

Any oracle whose concrete type `T <: AbstractOracle` implements the
required constructor interface can be used as the template.

`SeparableOracle` can also wrap explicitly constructed oracles, including
disjunctive oracles. A per-scenario split construction gives every scenario
its own one-dimensional DCGLP:

```julia
split_oracles = [
    begin
        kappa = ClassicalOracle(data, master; scen_idx = j)
        nu = ClassicalOracle(data, master; scen_idx = j)
        SplitOracle(
            master,
            (kappa, nu);
            param = deepcopy(split_param),
        )
    end
    for j in 1:N
]

oracle = SeparableOracle(master, split_oracles)
```

Each child declares how many local `t` values it consumes through
`auxiliary_dimension`. The wrapper assigns each child a contiguous block of
the global `t`, copies each returned cut, and embeds its local `a_t` into that
block, so child cut histories remain local. Scalar scenario oracles use blocks
of length one, while grouped oracles such as `UFLKnapsackOracle` may use larger
blocks.

### SUFLP with customer-disaggregated scenario blocks

For [`SUFLPData`](@ref), [`update_knapsack_master_model!`](@ref) creates
`t[customer, scenario]` and returns `vec(t)`. Julia's column-major ordering
therefore keeps all customers of one scenario in a contiguous block. A
scenario-aware `UFLKnapsackOracle` reports a block size equal to the number of
customers, and `SeparableOracle` combines those blocks automatically:

```julia
master = Master(data; model = update_knapsack_master_model!)
oracle = SeparableOracle(
    data,
    master,
    UFLKnapsackOracle,
    data.n_scenarios;
    sub_oracle_param = UFLKnapsackOracleParam(
        add_only_violated_cuts = true,
    ),
)
env = BendersSeq(master, oracle)
solve!(env)
```

Scenario probabilities appear only as coefficients of the master auxiliary
variables. The scenario subproblems and knapsack oracles return unweighted
recourse values, avoiding accidental double weighting. To use local
disjunctive separation, construct one `SplitOracle` from two scenario-specific
`UFLKnapsackOracle`s for each scenario, then wrap those split oracles in the
outer `SeparableOracle`.

!!! note
    `SeparableOracle` evaluates subproblems with Julia threads. The default GLPK
    optimizer is allowed, but GLPK may raise solver-internal errors during
    threaded subproblem evaluation on some platforms. If that occurs, pass a
    different LP optimizer with `optimizer = ...` or run Julia with one thread.

---

## Using Split Oracles
For mixed-integer master problems, BendersX provides
[`SplitOracle`](@ref) for generating **disjunctive Benders cuts** through a
Dual Cut Generating Linear Program (DCGLP).

A `SplitOracle` uses two *typical* oracles, denoted by `oracle_kappa` and
`oracle_nu`, together with a normalization scheme and a `SplitOracleParam`.
The typical oracles perform separation over the two sides of the split
disjunction, while the normalization scheme specifies the normalization
imposed in the DCGLP. `SplitOracleParam` controls the remaining algorithmic
choices and contains a [`DcglpParam`](@ref) for configuring the DCGLP.

```julia
using CPLEX

oracle_kappa = ClassicalOracle(data, master)
oracle_nu    = ClassicalOracle(data, master)

dcglp_optimizer = optimizer_with_attributes(
    CPLEX.Optimizer,
    "CPX_PARAM_EPRHS" => 1e-9,
    "CPX_PARAM_NUMERICALEMPHASIS" => 1,
    "CPX_PARAM_EPOPT" => 1e-9,
    MOI.Silent() => true,
)
dcglp_param = DcglpParam(dcglp_optimizer)
normalization = LpDistanceNormalization()
oracle_param = SplitOracleParam(;
    dcglp_param = dcglp_param,
    split_index_selection_rule = MostFractional(),
    strengthened = true,
    lift = true,
)
oracle = SplitOracle(
    master,
    (oracle_kappa, oracle_nu);
    normalization = normalization,
    param = oracle_param,
)
```
The DCGLP optimizer can be configured through the standard JuMP
`optimizer_with_attributes` interface, as shown above. `oracle_kappa` and `oracle_nu`
can be any typical oracles that are compatible with the subproblem.

These two composition orders have different meanings:

- `SplitOracle(Separable typical oracles)` builds one global DCGLP, so a cut
  may involve several `t[j]` components.
- `SeparableOracle(per-scenario SplitOracles)` builds one local DCGLP per
  scenario, so every generated cut acts only on that scenario's `t[j]`.

`SplitOracle` obtains its auxiliary-variable dimension from its component
oracles. Scalar typical oracles such as `ClassicalOracle` report dimension
one, while `SeparableOracle` reports the sum of its component-oracle
dimensions. The two component oracles must report the same dimension.

### Configuring `SplitOracle` Behavior

The normalization scheme is selected when constructing `SplitOracle`. The
default is `LpDistanceNormalization(Inf)`, which seeks a valid disjunctive cut
maximizing its $\ell_\infty$ distance from the separation point.
`LpDistanceNormalization(p)` provides an $\ell_p$-distance normalization for
$p \in \{1,2,\infty\}$, while `ReversePolarNormalization(...)` provides
reverse-polar normalization.

The remaining
behavior is controlled through `SplitOracleParam`. Key options include:
- Split selection
    - `split_index_selection_rule`: determines which fractional master variable is selected to form the disjunction.
- Cut management
    - `disjunctive_cut_append_rule`: controls how previously generated disjunctive cuts are reused.
    - `add_benders_cuts_to_master`: controls whether byproduct Benders cuts are added unconditionally, only when violated, or not at all.
- Strengthening and lifting
    - `strengthened`: enables strengthening of disjunctive cuts.
    - `lift`: applies lifting based on variables fixed to 0 or 1.
- DCGLP reuse
    - `reuse_dcglp`: reuses the DCGLP model from previous cut generation.
These options allow fine-grained control over performance and numerical
robustness.

---



## Choosing the Right Oracle

| Oracle type                                   | When to use it                                                              |
|----------------------------------------------|-----------------------------------------------------------------------------|
| `ClassicalOracle`, `UnifiedOracle`, `ParetoOracle` | General-purpose Benders decomposition                                      |
| `CFLKnapsackOracle`                           | Capacitated facility location problems                                      |
| `UFLKnapsackOracle`                           | Uncapacitated facility location problems                                    |
| `SplitOracle` with `LpDistanceNormalization` | General-purpose Benders decomposition for problems with an MILP master      |
| `SeparableOracle`                            | General-purpose Benders decomposition for problems with multi-scenario or separable recourse |
| Custom `AbstractOracle` | Research and prototyping |


---

## Summary

* Oracles are fully modular and interchangeable
* Behavior is controlled via dedicated parameter types
* Swapping oracles requires minimal code changes

This design enables rapid experimentation with different decomposition
strategies while preserving a consistent modeling interface.
