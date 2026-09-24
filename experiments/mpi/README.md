# MPI Examples

This directory contains examples of synchronous and asynchronous MPI-parallelized Benders decomposition using BendersX.jl.

## Examples

### `mpi_benders_synchronous.jl`

Synchronous MPI separation implemented through a user-defined Oracle. Rank 0 runs a built-in BendersX Environment, while partitioned `SeparableOracle`s on the MPI ranks evaluate the assigned subproblems. Rank 0 waits for all partitions before returning the global separation result.

### `mpi_benders_asynchronous.jl`

Asynchronous MPI execution implemented through a user-defined Environment. Worker results are processed as they arrive, and valid violated cuts can update the Master without waiting for all workers to finish earlier candidates.

In both examples, the component Oracles inside `SeparableOracle` can be any compatible BendersX Oracle.

## Requirements

The examples use:

- BendersX.jl
- MPI.jl
- JuMP
- a JuMP-compatible optimizer

MPI.jl can install its `mpiexecjl` wrapper with:

```julia
using MPI
MPI.install_mpiexecjl()
```

A separate system-wide MPI installation is not required when the MPI implementation supplied by MPI.jl is used.

CPLEX is recommended, particularly for multithreaded configurations and callback-based BendersX Environments. GLPK can be used for sequential or single-threaded configurations, but may fail when subproblems are evaluated concurrently using multiple Julia threads.

## Project environment

The MPI examples use a separate Julia environment. From this directory:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Running

Synchronous:

```bash
mpiexecjl -n 4 julia --project=. --threads=4 mpi_benders_synchronous.jl
```

Asynchronous:

```bash
mpiexecjl -n 4 julia --project=. --threads=4 mpi_benders_asynchronous.jl
```

An SCFLP instance name can be supplied as the first argument:

```bash
mpiexecjl -n 4 julia --project=. --threads=4 mpi_benders_asynchronous.jl f25-c50-s64-r10-2
```

For a detailed explanation of the architecture and the distinction between the two approaches, see the MPI Examples section of the BendersX documentation.

## Communication

The examples use serialized Julia objects for clarity. A production implementation could use typed MPI buffers, packed numerical data, and nonblocking communication.
