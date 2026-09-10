```@meta
CurrentModule = BendersX
```

# BendersX.jl Documentation

Welcome to the documentation for **BendersX.jl** — a modular, plug-and-play framework for Benders decomposition algorithms in Julia.

[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://github.com/asu-opt-lab/BendersX.jl/blob/main/LICENSE)



---

## Introduction

**BendersX.jl** is a modular and extensible framework for Benders decomposition algorithms in Julia. It is designed to support **both standard implementations and experimental extensions** in a unified, principled manner.

The name **BendersX** reflects the package philosophy: **“X” denotes both new application domains and extensions beyond classical Benders decomposition**, including alternative cut-generation strategies, stabilization mechanisms, and execution logic. The framework is designed to enable such applications and extensions to be implemented, combined, and evaluated with minimal friction through a plug-and-play architecture.

---

!!! tip "Quick overview"
    - **Components**: Master · Oracle (cut generator) · Environment (execution controller)
    - **Architecture**: JuMP-based modeling with a modular, hierarchical algorithmic framework
    - **Design goal**: Plug-and-play extensibility for rapid prototyping and reproducible experimentation

---

## [Quick start](@id quick-start)
### Installation
Install BendersX using Julia’s built-in package manager. Launch Julia and run the following commands at the `julia>` prompt:
```julia
import Pkg
Pkg.add("BendersX")
```

### Workflow
The following example illustrates a minimal workflow using BendersX:
```julia
using BendersX, JuMP

# 1. User-defined data
data = MyData(...)

# 2. Create master and provide the master model update
master = Master(data; model = update_master_model!)

# 3. Select oracle and provide the subproblem model update
oracle = ClassicalOracle(data, master; model = update_sub_model!)

# 4. Choose environment
env = BendersSeq(master, oracle)

# 5. Solve
log = solve!(env)

```
!!! info "Modeling"
    Users provide both the master and subproblem formulations through model-update functions written in standard JuMP syntax.
    See the [Modeling Interface](@ref modeling-interface) for concrete examples.
    If you are unfamiliar with JuMP, please refer to the JuMP.jl documentation for an introduction: [Julia JuMP](https://jump.dev/JuMP.jl/stable/).

!!! tip "Plug-and-Play"
    Oracle and Environment variants can be swapped freely, enabling rapid experimentation without changes to the formulation.

    **Built-in Oracle variants**

    | Oracle type        | Description |
    |--------------------|-----------------------|
    | [`ClassicalOracle`](@ref)  | Standard Benders cut generation based on classical dual information | 
    | `UnifiedOracle`    | Unified treatment of feasibility and optimality cuts | 
    | `ParetoOracle`     | Generates Pareto-optimal Benders cuts | 
    | [`SeparableOracle`](@ref)  | Wrapper oracle for separable subproblems | 
    | [`UFLKnapsackOracle`](@ref)  | Knapsack-based specialized oracle for uncapacitated facility location problems |
    | [`CFLKnapsackOracle`](@ref)  | Knapsack-based specialized oracle for capacitated facility location problems |
    | [`SplitOracle`](@ref)      | Generic split-cut oracle with a configurable DCGLP normalization scheme |
    | `SplitOracle` with `LpDistanceNormalization` | Distance-norm split oracle configuration |

    **Built-in Environment variants**

    | Environment type        | Execution mode |
    |-------------------------|--------------------|
    | [`BendersSeq`](@ref)            | Classical sequential Benders decomposition |
    | [`BendersSeqInOut`](@ref)       | Sequential Benders with in–out stabilization |
    | [`BendersBnB`](@ref)            | Branch-and-bound integrated with Benders cuts |

## Key features
- **Plug-and-play architecture**: easily swap or extend Master, Oracle, and Environment components
- **JuMP-native modeling**: master and subproblems are expressed entirely using standard JuMP syntax
- **Built-in algorithmic variants**: classical, unified, Pareto, split cuts, in–out stabilization, branch-and-bound, and more
- **Benchmark-ready**: facility location, network interdiction, and additional problems for reproducible comparisons

!!! info "Next steps"
    - Explore **Tutorials** for worked examples and step-by-step guides
    - See the **User Guide** for advanced usage patterns
    - Consult the **API** for detailed documentation of Master, Oracle, and Environment interfaces
