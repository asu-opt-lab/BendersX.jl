# MPI Examples

BendersX.jl can be extended with MPI-based distributed execution without modifying its core Master, Oracle, or Environment interfaces. The examples in this directory illustrate two ways to introduce MPI parallelism at different levels of the framework.

## Synchronous MPI separation

[`mpi_benders_synchronous.jl`](https://github.com/asu-opt-lab/BendersX.jl/blob/main/experiments/mpi/mpi_benders_synchronous.jl) demonstrates synchronous distributed separation through a user-defined Oracle.

Rank 0 owns the evolving Benders Master and runs a built-in BendersX Environment, such as `BendersSeq`. The MPI Oracle distributes each candidate to the participating MPI ranks, where partitioned `SeparableOracle`s separate their assigned subproblems. Rank 0 waits for all partitions to finish, assembles the results, and returns them through the ordinary `generate_cuts` interface.

The Environment therefore remains unchanged. This example illustrates how distributed separation can be introduced by extending the Oracle layer while reusing the existing Environment and Master implementations.

## Asynchronous MPI separation

[`mpi_benders_asynchronous.jl`](https://github.com/asu-opt-lab/BendersX.jl/blob/main/experiments/mpi/mpi_benders_asynchronous.jl) demonstrates asynchronous distributed execution through a user-defined Environment.

Rank 0 maintains and solves the evolving Master, while worker ranks evaluate partitioned `SeparableOracle`s independently. Worker results are processed as they arrive. Cuts that are violated by the latest Master candidate can be incorporated immediately, and the Master can be reoptimized while other workers are still processing earlier candidates.

Each worker processes one candidate at a time. When a worker becomes available, it receives the latest Master candidate rather than waiting for the other workers to finish.

This example illustrates how a new execution strategy can be implemented by extending the Environment layer while retaining the ordinary BendersX Master and Oracle interfaces.

## Synchronous vs. asynchronous execution

The two examples differ primarily in where MPI coordination is introduced:

| | Synchronous | Asynchronous |
|---|---|---|
| MPI extension | Oracle | Environment |
| Master | Rank 0 | Rank 0 |
| Worker separation | `SeparableOracle` | `SeparableOracle` |
| Component Oracles | Any compatible Oracle | Any compatible Oracle |
| Candidate processing | Wait for all partitions | Process results as they arrive |
| Master update | After complete separation | As valid cuts arrive |
| Worker reassignment | Next common candidate | Latest available candidate |
| Built-in Environment reusable | Yes | Replaced by asynchronous Environment |

In both examples, a `SeparableOracle` may contain any compatible Oracle type. The `SeparableOracle` is responsible for organizing the assigned subproblems and their auxiliary-variable mappings; the choice of component Oracle determines the separation procedure.

## Partitioning and auxiliary variables

The MPI examples partition the independent subproblems across MPI processes. Each process receives a flat set of global `subproblem_indices`, while the corresponding `SeparableOracle` retains the component-level grouping.

Generated cuts are represented in the global auxiliary space. Objective values returned by a partitioned `SeparableOracle` correspond to the represented auxiliary variables and are assembled using `auxiliary_indices`.

Thus, the MPI layer does not introduce a separate local auxiliary-space convention. The ordinary `SeparableOracle` handles the local-to-global mapping, and the MPI layer assembles the results in the global auxiliary space.

## Running the examples

The examples use [`MPI.jl`](https://juliaparallel.org/MPI.jl/). MPI.jl can install the `mpiexecjl` wrapper for the MPI implementation used by the Julia environment:

```julia
using MPI
MPI.install_mpiexecjl()
```

From the MPI examples directory, instantiate the example environment and run, for example:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
mpiexecjl -n 4 julia --project=. --threads=4 mpi_benders_synchronous.jl
```

For the asynchronous example:

```bash
mpiexecjl -n 4 julia --project=. --threads=4 mpi_benders_asynchronous.jl
```

An SCFLP instance name can be supplied as the first command-line argument:

```bash
mpiexecjl -n 4 julia --project=. --threads=4 mpi_benders_asynchronous.jl f25-c50-s64-r10-2
```

## Solver considerations

CPLEX is recommended, particularly for multithreaded configurations and for BendersX Environments that rely on callback functionality.

GLPK can be used for sequential or single-threaded configurations, but it may fail when local subproblems are evaluated concurrently using multiple Julia threads.

## Implementation notes

MPI remains outside the BendersX package in both examples. Communication is implemented by user-defined code, while the BendersX Master and Oracle interfaces remain unchanged.

For clarity, the examples communicate serialized Julia objects through MPI. A production implementation could instead use typed MPI buffers, packed numerical data, and nonblocking communication.
