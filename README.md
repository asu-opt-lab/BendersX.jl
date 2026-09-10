# BendersX.jl

Welcome to **BendersX.jl** — a modular, plug-and-play framework for Benders decomposition algorithms in Julia.

[![Julia](https://img.shields.io/badge/julia-v1.11%2B-blue.svg)](https://julialang.org/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-source-blue.svg)](docs/)

---

## Introduction

**BendersX.jl** is a modular and extensible framework for Benders decomposition in Julia. It supports both standard implementations and experimental extensions within a unified, principled design.

The **“X”** in BendersX signals extension and exploration: new application domains and methodological extensions beyond classical Benders decomposition — including alternative cut-generation strategies, stabilization mechanisms, and execution logic. The framework’s plug-and-play architecture makes it easy to implement, combine, and evaluate these techniques with minimal boilerplate.

---

> **Quick overview**
>
> - **Components:** Master · Oracle (cut generator) · Environment (execution controller)
> - **Architecture:** JuMP-based modeling with a modular, hierarchical algorithmic framework
> - **Design goal:** Plug-and-play extensibility for rapid prototyping and reproducible experimentation
> - **Repository layout:** package source in `src/`, docs in `docs/`, and benchmark scripts in `experiments/`

---

## Installation

Install BendersX with the Julia package manager.

To add the package from GitHub:
```julia
import Pkg
Pkg.add(url = "https://github.com/asu-opt-lab/BendersX.jl.git")
```

If you are developing the repository locally:
```julia
import Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Quick start

### Minimal workflow

This example shows a minimal end-to-end workflow with a concrete optimizer and explicit model-update functions.

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

### Modeling

Users describe a decomposition model by writing ordinary JuMP code in two functions, one for the master and another for the subproblem. The master function adds the first-stage variables, constraints, and objective, then returns `(x, t)`, where `x` is a named tuple of master variables and `t` contains the auxiliary variables used in Benders cuts. A model-based oracle uses a subproblem function that adds the recourse variables and constraints; its keyword arguments must match the names returned in `x`.

Pass these functions to `Master` and the oracle through the `model` keyword. Problem data can be any Julia object. If a required modeling function is missing for its type, the default method throws `UnimplementedInterfaceException`. See the package documentation for complete examples. If you are new to JuMP, start with the JuMP documentation: [https://jump.dev/JuMP.jl/stable/](https://jump.dev/JuMP.jl/stable/).

## Built-in variants

### Oracle variants (examples)

| Oracle type         | Description                                                          |
| ------------------- | -------------------------------------------------------------------- |
| `ClassicalOracle`   | Standard Benders cut generation based on dual information            |
| `UnifiedOracle`     | Unified handling of feasibility and optimality cuts                  |
| `ParetoOracle`      | Produces Pareto-optimal Benders cuts                                 |
| `SeparableOracle`   | Wrapper oracle for separable subproblems                             |
| `UFLKnapsackOracle` | Knapsack-based oracle for uncapacitated facility location            |
| `CFLKnapsackOracle` | Knapsack-based oracle for capacitated facility location              |
| `SplitOracle`       | Generates split cuts to strengthen the master relaxation             |

`ClassicalOracle`, `UnifiedOracle`, and `ParetoOracle` are general model-based oracles. `SeparableOracle`, `SplitOracle`, `UFLKnapsackOracle`, and `CFLKnapsackOracle` are specialized constructors with different signatures and setup requirements; refer to the corresponding docs/API before swapping them into a workflow.

### Environment variants (examples)

| Environment type        | Execution strategy                                |
| ----------------------- | ------------------------------------------------- |
| `BendersSeq`            | Classical sequential Benders decomposition        |
| `BendersSeqInOut`       | Sequential Benders with in-out stabilization      |
| `BendersBnB`            | Branch-and-bound integrated with Benders cuts     |

> These are representative built-ins. See the documentation for the full list and configuration options.

---

## Repository layout

This repository contains the package source code, benchmark loaders, documentation, and computational experiments for `BendersX`.

- `src/BendersX.jl` centralizes the package public API and exports
- `src/modules/` contains masters, environments, oracles, and callback logic
- `src/problems/` contains benchmark readers, problem models, and specialized problem-specific oracles
- `src/artifact_utils.jl` and `Artifacts.toml` manage artifact-backed benchmark datasets downloaded on demand
- `docs/` contains the standalone Documenter site, tutorials, and API pages
- `experiments/` contains experiment scripts and reference objectives used for benchmarking
- `test/` contains public API and unit tests

---

## Key features

- **Plug-and-play architecture:** swap or extend Master, Oracle, and Environment components without rewriting models.
- **JuMP-native modeling:** define master and subproblems using standard JuMP expressions and containers.
- **Algorithmic variants:** support for classical, unified, Pareto, split cuts, in-out stabilization, branch-and-bound integration, and more.
- **Artifact-backed benchmark data:** built-in readers for CFLP, UFLP, SCFLP, and SNIP obtain packaged datasets lazily when needed.
- **Benchmark-ready workflow:** tutorials, docs, and experiment suites are organized for reproducible experimentation.

---

## Testing and docs

Run the test suite:

```bash
julia --project=. test/runtests.jl
```

Build the documentation locally:

```bash
julia --project=docs docs/make.jl
```

Benchmark and algorithm comparison scripts live under `experiments/`, including sequential, in-out, callback, and disjunctive experiment setups.

---

## Next steps

- Refer to the **Tutorials** documentation for worked examples and step-by-step guides.
- Refer to the **User Guide** for advanced usage patterns and customization guidance.
- Consult the **API** documentation for detailed descriptions of the Master, Oracle, and Environment interfaces.

## Contributing

Contributions, bug reports, and enhancements are welcome. Please open issues or pull requests on GitHub. Follow the repository's coding guidelines and run the test suite locally before submitting PRs.

---

## License

BendersX.jl is released under the **MIT License**. See [LICENSE](LICENSE) for details.
