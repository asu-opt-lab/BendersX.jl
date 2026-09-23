# Note: 
# GLPK raises memory error when nthreads > 1. 
# GLPK does not provide MOI.RelativeGapTolerance()
# SeparableOracle:
#     one component -> one or more subproblems

# MPI partition:
#     one process -> a flat set of assigned subproblems
"""
    mpi_benders_synchronous.jl

Synchronous MPI-parallelized Benders separation using BendersX.jl.

MPI remains outside the BendersX package. A user-defined Oracle wrapper
distributes candidate separation across MPI ranks, while each rank uses the
built-in `SeparableOracle` to separate its assigned subproblems. The
surrounding BendersX Environment, such as `BendersSeq`, remains unchanged.

Architecture
------------
- Rank 0 owns the evolving Benders Master and runs BendersX Environment, `BendersSeq` in this example.
- Each rank constructs a local BendersX `Master` only to construct its
  `SeparableOracle` and component Oracles. Worker-side Masters are not updated
  during the Benders solve.
- The independent subproblems are partitioned across MPI ranks. Each rank owns
  a `SeparableOracle` representing its assigned subproblems.
- Rank 0 uses `MPISynchronousSeparableOracle` as its Oracle. Its
  `generate_cuts` method distributes each candidate to the workers, evaluates
  the local partition, collects the worker results, and assembles the global
  separation result.
- Worker ranks remain in a service loop and evaluate only their assigned
  subproblems.

Solver
------
This example recommends CPLEX, which supports the robust solver-independent
functionality used by the BendersX Environment.

GLPK can be used for sequential BendersX Environments, but it may fail when
subproblems are evaluated concurrently using multiple Julia threads. It is
not recommended for callback-based Environments such as `BendersBnB`.

Run, for example:

    mpiexecjl -n 4 julia --threads=4 mpi_benders_synchronous.jl

or with another SCFLP instance:

    mpiexecjl -n 4 julia --threads=4 mpi_benders_synchronous.jl f25-c50-s64-r10-2

For clarity, this example communicates serialized Julia objects through MPI.
A production implementation could instead use typed MPI buffers and
nonblocking communication for numerical arrays and packed cut data.
"""

using MPI
using JuMP
using BendersX

import BendersX: AbstractOracle, Hyperplane, generate_cuts

if !isnothing(Base.find_package("CPLEX"))
    @eval using CPLEX
    optimizer = CPLEX.Optimizer
else
    @warn "CPLEX is recommended for multithreaded and callback-based configurations; falling back to GLPK."
    @eval using GLPK
    optimizer = GLPK.Optimizer
end

# -----------------------------------------------------------------------------
# MPI setup
# -----------------------------------------------------------------------------

# MPI calls in this example are made only by the main Julia thread. Local
# SeparableOracle computation may use additional Julia threads.
MPI.Init(threadlevel = :funneled)

const COMM = MPI.COMM_WORLD
const ROOT = 0
const TAG_PARTITION = 100
const TAG_REQUEST = 200
const TAG_RESULT = 300
const TAG_STOP = 400

rank = MPI.Comm_rank(COMM)
nranks = MPI.Comm_size(COMM)

# -----------------------------------------------------------------------------
# Problem data
# -----------------------------------------------------------------------------

# Every process constructs the same problem data locally. This avoids sending
# JuMP models or other solver objects through MPI.
instance_name = length(ARGS) >= 1 ? ARGS[1] : "f25-c50-s64-r10-1"
data = read_stochastic_capacited_facility_location_problem(instance_name)
N = data.n_scenarios

# -----------------------------------------------------------------------------
# Global partitioning
# -----------------------------------------------------------------------------

"""
    balanced_partitions(N, nranks)

Partition `1:N` into `nranks` contiguous, nearly equal-sized subproblem index
sets. The returned vector is indexed by MPI rank + 1.
"""
function balanced_partitions(N::Int, nranks::Int)
    nranks > 0 || throw(ArgumentError("nranks must be positive"))
    nranks <= N || throw(ArgumentError(
        "number of MPI ranks ($nranks) cannot exceed the number of subproblems ($N)",
    ))

    q, r = divrem(N, nranks)
    partitions = Vector{Vector{Int}}(undef, nranks)

    first_idx = 1
    for rank in 1:nranks
        n = q + (rank <= r ? 1 : 0)
        last_idx = first_idx + n - 1
        partitions[rank] = collect(first_idx:last_idx)
        first_idx = last_idx + 1
    end

    return partitions
end

# Rank 0 computes the partition and sends one partition to each worker.
if rank == ROOT
    partitions = balanced_partitions(N, nranks)

    for worker in 1:(nranks - 1)
        MPI.send(
            partitions[worker + 1],
            COMM;
            dest = worker,
            tag = TAG_PARTITION,
        )
    end
else
    subproblem_indices = MPI.recv(
        COMM;
        source = ROOT,
        tag = TAG_PARTITION,
    )
end

MPI.Barrier(COMM)

if rank == ROOT
    subproblem_indices = partitions[1]
end

for r in 0:(nranks - 1)
    MPI.Barrier(COMM)
    if rank == r
        println("Rank $rank: assigned subproblems $subproblem_indices")
    end
end
MPI.Barrier(COMM)

# -----------------------------------------------------------------------------
# Local construction-only Master and partitioned SeparableOracle
# -----------------------------------------------------------------------------

# This Master is needed only because the BendersX Oracles use Master
# information during construction. On worker ranks it is not the evolving
# Benders master and does not receive generated cuts.
construction_master = Master(data)

local_N = length(subproblem_indices)
local_oracle = SeparableOracle(
    data,
    construction_master,
    ClassicalOracle,
    local_N;
    subproblem_indices = subproblem_indices,
    optimizer = optimizer_with_attributes(optimizer, MOI.Silent() => true),
)

# -----------------------------------------------------------------------------
# MPI Oracle wrapper
# -----------------------------------------------------------------------------

"""
    MPISynchronousSeparableOracle <: AbstractOracle

User-defined Oracle that coordinates partitioned `SeparableOracle`s across MPI
ranks while preserving the ordinary synchronous `generate_cuts` interface.

Only the root rank calls this Oracle's `generate_cuts` method. Worker ranks
run `serve_worker!` below.
"""
struct MPISynchronousSeparableOracle <: AbstractOracle
    local_oracle::SeparableOracle
    comm::MPI.Comm
    root::Int
    workers::Vector{Int}
    partitions::Vector{Vector{Int}}
end

"""
    assemble_results(results, partitions, N)

Assemble local partition results into a global separation result on the root.
"""
function assemble_results(
    results::Dict{Int,Any},
    partitions::Vector{Vector{Int}},
    dim_global_auxiliary::Int,
)
    global_sub_obj_vals = Vector{Float64}(undef, dim_global_auxiliary)
    global_cuts = Hyperplane[]
    is_in_L = true

    for rank in sort!(collect(keys(results)))
        result = results[rank]
        expected_subproblem_indices = partitions[rank + 1]

        result.subproblem_indices == expected_subproblem_indices ||
            throw(ArgumentError(
                "MPI separation result from rank $rank has subproblem indices " *
                "$(result.subproblem_indices), expected " *
                "$expected_subproblem_indices.",
            ))

        local_auxiliary_indices = vcat(result.auxiliary_indices...)

        length(result.sub_obj_vals) == length(local_auxiliary_indices) ||
            throw(DimensionMismatch(
                "MPI separation result from rank $rank contains " *
                "$(length(result.sub_obj_vals)) objective values for " *
                "$(length(local_auxiliary_indices)) represented auxiliary variables.",
            ))

        is_in_L &= result.is_in_L

        for k in eachindex(local_auxiliary_indices)
            global_sub_obj_vals[local_auxiliary_indices[k]] =
                result.sub_obj_vals[k]
        end

        # Hyperplanes are already expressed in the global (x, t) space.
        append!(global_cuts, result.cuts)
    end

    return is_in_L, global_cuts, global_sub_obj_vals
end

"""
    generate_cuts(::MPISynchronousSeparableOracle, x_value, t_value; ...)

Distribute one complete separation request and synchronously wait for every
partition to finish before returning the global separation result.
"""
function generate_cuts(
    oracle::MPISynchronousSeparableOracle,
    x_value::Vector{Float64},
    t_value::Vector{Float64};
    tol_normalize = 1.0,
    time_limit = 3600.0,
)
    MPI.Comm_rank(oracle.comm) == oracle.root ||
        throw(ArgumentError(
            "MPISynchronousSeparableOracle.generate_cuts must be called " *
            "on the root rank."
        ))

    request = (
        x = x_value,
        t = t_value,
        tol_normalize = tol_normalize,
        time_limit = time_limit,
    )

    # Dispatch the same candidate to every worker.
    for worker in oracle.workers
        MPI.send(
            (command = :separate, request = request),
            oracle.comm;
            dest = worker,
            tag = TAG_REQUEST,
        )
    end

    # Rank 0 evaluates its own partition through the ordinary BendersX
    # `generate_cuts` interface.
    root_is_in_L, root_cuts, root_sub_obj_vals = generate_cuts(
        oracle.local_oracle,
        x_value,
        t_value;
        tol_normalize = tol_normalize,
        time_limit = time_limit,
    )

    results = Dict{Int,Any}(
        oracle.root => (
            subproblem_indices = vcat(oracle.local_oracle.subproblem_indices...),
            auxiliary_indices = copy(oracle.local_oracle.auxiliary_indices),
            is_in_L = root_is_in_L,
            cuts = root_cuts,
            sub_obj_vals = root_sub_obj_vals,
        ),
    )

    # Synchronous interface: wait until every worker has returned.
    for worker in oracle.workers
        message = MPI.recv(
            oracle.comm;
            source = worker,
            tag = TAG_RESULT,
        )

        message.command === :result ||
            throw(ArgumentError("Unexpected MPI message from rank $worker."))

        results[worker] = message.result
    end

    return assemble_results(
        results,
        oracle.partitions,
        oracle.local_oracle.dim_global_auxiliary,
    )
end

# -----------------------------------------------------------------------------
# Worker service loop
# -----------------------------------------------------------------------------

function serve_worker!(local_oracle::SeparableOracle, comm::MPI.Comm; root = ROOT)
    while true
        message = MPI.recv(
            comm;
            source = root,
            tag = MPI.ANY_TAG,
        )

        command = get(message, :command, nothing)

        if command === :stop
            return
        elseif command === :separate
            request = message.request

            # Workers also use the ordinary BendersX `generate_cuts` interface;
            # no MPI-specific local-separation API is required.
            local_is_in_L, local_cuts, local_sub_obj_vals = generate_cuts(
                local_oracle,
                request.x,
                request.t;
                tol_normalize = request.tol_normalize,
                time_limit = request.time_limit,
            )

            result = (
                    subproblem_indices = vcat(local_oracle.subproblem_indices...),
                    auxiliary_indices = copy(local_oracle.auxiliary_indices),
                    is_in_L = local_is_in_L,
                    cuts = local_cuts,
                    sub_obj_vals = local_sub_obj_vals,
            )

            MPI.send(
                (command = :result, result = result),
                comm;
                dest = root,
                tag = TAG_RESULT,
            )
        else
            throw(ArgumentError("Unknown command received by worker."))
        end
    end
end

# -----------------------------------------------------------------------------
# Root / worker roles
# -----------------------------------------------------------------------------

if rank == ROOT
    # The authoritative, evolving Master exists only on rank 0.
    master = Master(data; optimizer = optimizer_with_attributes(optimizer, MOI.Silent() => true))

    # The MPI wrapper is a user-defined Oracle. BendersSeq does not need to
    # know that generate_cuts performs distributed computation internally.
    mpi_oracle = MPISynchronousSeparableOracle(
        local_oracle,
        COMM,
        ROOT,
        collect(1:(nranks - 1)),
        partitions,
    )

    preprocessing = LPRelaxationPreprocessing(mpi_oracle; seq_env_type = BendersSeq, param = BendersSeqParam(;
        time_limit = 200.0,
        gap_tolerance = 1e-9
    ))

    # env = BendersSeq(master, mpi_oracle; preprocessing = preprocessing)
    # env = BendersBnB(master, mpi_oracle; preprocessing = preprocessing)
    env = BendersSeqInOut(master, mpi_oracle; preprocessing = preprocessing, param = BendersSeqInOutParam(stabilizing_x = ones(data.n_facilities)))

    try
        solve!(env)

        println("rank 0: termination = $(env.termination_status)")
        println("rank 0: objective   = $(env.obj_value)")
    finally
        # Always release worker service loops, including when solve! throws.
        stop_command = (command = :stop,)
        for worker in 1:(nranks - 1)
            MPI.send(
                stop_command,
                COMM;
                dest = worker,
                tag = TAG_STOP,
            )
        end
    end
else
    # Workers never construct or run a Benders Environment.
    serve_worker!(local_oracle, COMM; root = ROOT)
end

MPI.Barrier(COMM)
MPI.Finalize()
