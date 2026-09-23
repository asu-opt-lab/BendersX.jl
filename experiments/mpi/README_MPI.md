# MPI Examples

These examples illustrate how MPI-based distributed execution can be
added to BendersX.jl without modifying its core Master, Oracle, or
Environment interfaces.

The examples partition independent subproblems across MPI processes and
use the built-in `SeparableOracle` to perform separation on each
partition.

## Examples

### `mpi_benders_synchronous.jl`

Demonstrates synchronous distributed separation through a user-defined
MPI-aware Oracle.

Rank 0 runs an ordinary BendersX Environment. The MPI Oracle distributes
each candidate across the participating ranks, waits for all partitions
to finish, assembles their results, and returns through the standard
`generate_cuts` interface.

This design leaves the BendersX Environment unchanged and illustrates
how distributed separation can be implemented by extending the Oracle
layer.

### `mpi_benders_asynchronous.jl`

Demonstrates asynchronous distributed separation through a user-defined
Environment.

Rank 0 maintains and solves the evolving Master, while worker ranks
evaluate partitioned `SeparableOracle`s independently. Worker results
are processed as they arrive. Valid cuts can therefore update the Master
without waiting for the other workers to finish evaluating earlier
candidates.

When a worker becomes available, it receives the latest Master
candidate.

This design illustrates how a new execution strategy can be implemented
by extending the Environment layer while retaining the ordinary BendersX
Master and Oracle interfaces.

## Requirements

The examples require:

-   MPI.jl
-   an MPI implementation configured for MPI.jl
-   BendersX.jl
-   a JuMP-compatible optimizer

CPLEX is recommended, particularly when using multiple Julia threads or
callback-based BendersX Environments. GLPK can be used for sequential or
single-threaded configurations, but may fail when subproblems are
evaluated concurrently using multiple Julia threads.

MPI.jl can install an `mpiexecjl` wrapper corresponding to its
configured MPI implementation:

``` julia
using MPI
MPI.install_mpiexecjl()
```

## Running the examples

For example, run the synchronous implementation with four MPI processes
and four Julia threads per process:

``` bash
mpiexecjl -n 4 julia --threads=4 mpi_benders_synchronous.jl
```

Run the asynchronous implementation similarly:

``` bash
mpiexecjl -n 4 julia --threads=4 mpi_benders_asynchronous.jl
```

An SCFLP instance name can be supplied as the first command-line
argument:

``` bash
mpiexecjl -n 4 julia --threads=4 mpi_benders_asynchronous.jl \
    f25-c50-s64-r10-2
```

## Synchronous vs. asynchronous execution

The two examples differ primarily in where MPI coordination is
introduced.

  -----------------------------------------------------------------------
                          Synchronous             Asynchronous
  ----------------------- ----------------------- -----------------------
  MPI extension           Oracle                  Environment

  Master                  Rank 0                  Rank 0

  Worker separation       `SeparableOracle`       `SeparableOracle`

  Candidate processing    Wait for all partitions Process results as they
                                                  arrive

  Master update           After complete          As valid cuts arrive
                          separation              

  Worker reassignment     Next common candidate   Latest available
                                                  candidate

  Built-in Environment    Yes                     Replaced by
  reusable                                        asynchronous
                                                  Environment
  -----------------------------------------------------------------------

Both examples use serialized Julia objects for MPI communication to keep
the implementation readable. A production implementation may instead use
typed MPI buffers, packed cut representations, and nonblocking
communication.
