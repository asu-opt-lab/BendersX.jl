"""
    mpi_benders_asynchronous.jl

Asynchronous MPI-parallelized Benders separation using BendersX.jl.

MPI remains outside the BendersX package. A user-defined Environment
coordinates an evolving Master on rank 0 with partitioned `SeparableOracle`s on
worker ranks. Workers use the ordinary BendersX `generate_cuts` interface, so
asynchronous coordination is introduced without changing the Master or Oracle
interfaces.

Architecture
------------
- Rank 0 owns the evolving Benders Master and runs the asynchronous Benders
  Environment.
- Worker ranks construct local `SeparableOracle`s representing disjoint sets
  of global subproblems. Their local Masters are used only to construct the
  Oracles and are not updated during the solve.
- Each worker processes one candidate at a time. When a worker becomes
  available, it receives the latest Master candidate rather than waiting for
  other workers to finish.
- Rank 0 processes worker results as they arrive. Cuts are incorporated as
  soon as they are received, and the Master may be reoptimized while other
  workers are still processing earlier candidates.
- The objective value of every Master candidate provides a valid lower bound.
  A candidate can provide an upper-bound update once all workers have
  evaluated it and its auxiliary values are available.

Solver
------
CPLEX is recommended for this example. GLPK can be used for sequential or
single-threaded configurations, but it may fail when subproblems are evaluated
concurrently using multiple Julia threads.

Run, for example:

    mpiexecjl -n 4 julia --threads=4 mpi_benders_asynchronous.jl

or with another SCFLP instance:

    mpiexecjl -n 4 julia --threads=4 mpi_benders_asynchronous.jl f25-c50-s64-r10-2

For clarity, this example communicates serialized Julia objects through MPI.
A production implementation could instead use typed MPI buffers and
nonblocking communication for numerical arrays and packed cut data.
"""

using MPI
using JuMP
using Printf
using BendersX
import MathOptInterface as MOI

import BendersX: AbstractBendersEnv, AbstractMaster, AbstractOracle
import BendersX: AbstractBendersSeqState, AbstractBendersSeqLog, AbstractBendersSeqParam
import BendersX: get_sec_remaining, evaluate_violation
import BendersX: generate_cuts, master_model
import BendersX: linking_variables, auxiliary_variables, evaluate_objective
import BendersX: add_cuts!, is_solved_and_feasible, termination_status


# -----------------------------------------------------------------------------
# Candidate and worker result types
# -----------------------------------------------------------------------------

struct Candidate
    id::Int
    values::Dict{Symbol, Vector{Float64}}
    master_value::Float64
end

mutable struct CandidateState
    candidate::Candidate
    completed_workers::Set{Int}
    is_in_L::Dict{Int,Bool}
    sub_obj_vals::Vector{Float64}

    function CandidateState(candidate::Candidate)
        new(
            candidate,
            Set{Int}(),
            Dict{Int,Bool}(),
            fill(NaN, length(candidate.values[:t])),
        )
    end
end

function fully_evaluated(
    state::CandidateState,
    workers::Vector{Int},
)
    length(state.completed_workers) == length(workers)
end

mutable struct Log <: AbstractBendersSeqLog
    LB::Float64
    UB::Float64
    gap::Float64
    best_sol_id::Int
    states::Dict{Int,CandidateState}
    busy_workers::Set{Int}
    start_time::Float64
    consecutive_no_improvement::Int
    
    function Log()
        new(-Inf, Inf, 100.0, -1, Dict{Int,CandidateState}(), Set{Int}(), time(), 0)
    end
end

mutable struct Param <: AbstractBendersSeqParam

    time_limit::Float64
    gap_tolerance::Float64
    verbose::Bool

    function Param(; 
                        time_limit::Float64 = 7200.0, 
                        gap_tolerance::Float64 = 1e-6, 
                        verbose::Bool = true
                        ) 
        
        new(time_limit, gap_tolerance, verbose)
    end
end

function update_upper_bound_and_gap!(
    state::CandidateState,
    log::Log,
    f::Function;
    zero_tol::Float64 = 1e-9,
)
    evaluation = f(
        max.(state.candidate.values[:t], state.sub_obj_vals),
        state.candidate.values[:x],
    )

    if evaluation < log.UB 
        log.best_sol_id = state.candidate.id
        log.UB = evaluation
    end

    if isapprox(log.UB, log.LB; atol = zero_tol)
        log.gap = 0.0
    elseif abs(log.UB) <= zero_tol
        log.gap = Inf
    else
        log.gap = max(
            0.0,
            (log.UB - log.LB) / abs(log.UB),
        )
    end

    return nothing
end

function print_iteration_info(state::CandidateState, log::Log; prefix="")
    @printf("%s Id: %4d | LB: %12.4f | UB: %11.4f | Gap: %8.3f%%) \n",
           prefix, state.candidate.id, log.LB, log.UB, log.gap * 100)
end
# -----------------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------------

if !isnothing(Base.find_package("CPLEX"))
    @eval using CPLEX
    optimizer = CPLEX.Optimizer
else
    @warn "CPLEX is recommended for this example; falling back to GLPK. GLPK may fail when local subproblems are evaluated concurrently using multiple Julia threads."
    @eval using GLPK
    optimizer = GLPK.Optimizer
end

# -----------------------------------------------------------------------------
# MPI setup
# -----------------------------------------------------------------------------

MPI.Init(threadlevel = :funneled)

const COMM = MPI.COMM_WORLD
const ROOT = 0
const TAG_PARTITION = 100
const TAG_WORK = 200
const TAG_RESULT = 300
const TAG_STOP = 400

rank = MPI.Comm_rank(COMM)
nranks = MPI.Comm_size(COMM)

nranks >= 2 || error("This asynchronous example requires at least two MPI ranks.")

# -----------------------------------------------------------------------------
# Problem data
# -----------------------------------------------------------------------------

instance_name = length(ARGS) >= 1 ? ARGS[1] : "f25-c50-s64-r10-1"
data = read_stochastic_capacited_facility_location_problem(instance_name)
N = data.n_scenarios

# -----------------------------------------------------------------------------
# Global partitioning
# -----------------------------------------------------------------------------

"""
    balanced_partitions(N, nworkers)

Partition `1:N` into `nworkers` contiguous, nearly equal-sized subproblem
index sets. The returned vector is indexed by worker index.
"""
function balanced_partitions(N::Int, nworkers::Int)
    nworkers > 0 || throw(ArgumentError("nworkers must be positive"))
    nworkers <= N || throw(ArgumentError(
        "number of MPI workers ($nworkers) cannot exceed the number of subproblems ($N)",
    ))

    q, r = divrem(N, nworkers)
    partitions = Vector{Vector{Int}}(undef, nworkers)

    first_idx = 1
    for worker_idx in 1:nworkers
        n = q + (worker_idx <= r ? 1 : 0)
        last_idx = first_idx + n - 1
        partitions[worker_idx] = collect(first_idx:last_idx)
        first_idx = last_idx + 1
    end

    return partitions
end

if rank == ROOT
    worker_partitions = balanced_partitions(N, nranks - 1)
    for worker in 1:(nranks - 1)
        MPI.send(
            worker_partitions[worker],
            COMM;
            dest = worker,
            tag = TAG_PARTITION,
        )
    end
    subproblem_indices = Int[]
else
    subproblem_indices = MPI.recv(
        COMM;
        source = ROOT,
        tag = TAG_PARTITION,
    )
end

MPI.Barrier(COMM)

# Display worker ownership in rank order.
for worker in 1:(nranks - 1)
    MPI.Barrier(COMM)
    if rank == worker
        println("Rank $rank: assigned subproblems $subproblem_indices")
    end
end
MPI.Barrier(COMM)

# -----------------------------------------------------------------------------
# Local partitioned SeparableOracle on workers
# -----------------------------------------------------------------------------

if rank != ROOT
    # This Master is used only because BendersX Oracles use Master information
    # during construction. It is not the evolving Benders Master.
    construction_master = Master(data)

    local_oracle = SeparableOracle(
        data,
        construction_master,
        CFLKnapsackOracle,
        length(subproblem_indices);
        subproblem_indices = subproblem_indices,
        optimizer = optimizer_with_attributes(
            optimizer,
            MOI.Silent() => true,
        ),
    )
end


# -----------------------------------------------------------------------------
# Asynchronous Environment
# -----------------------------------------------------------------------------

"""
    MPIAsynchronousBendersEnv <: AbstractBendersEnv

Asynchronous Benders Environment in which workers process candidate versions
independently. Cuts returned for older candidates remain valid and are
incorporated when they arrive. A candidate provides a lower bound once all
workers have evaluated it, and is optimal when all workers report membership
in `L`.
"""
mutable struct MPIAsynchronousBendersEnv <: AbstractBendersEnv
    master::AbstractMaster
    comm::MPI.Comm
    workers::Vector{Int}
    partitions::Vector{Vector{Int}}

    param::Param

    termination_status::BendersX.TerminationStatus
    obj_value::Float64
end

function master_candidate(
    env::MPIAsynchronousBendersEnv,
    id::Int,
)
    model = master_model(env.master)
    optimize!(model)

    is_solved_and_feasible(model; allow_local = false, dual = false) || throw(
        BendersX.UnexpectedModelStatusException(
            "MPIAsynchronousBendersEnv: master has status " *
            "$(termination_status(model)).",
        ),
    )

    x = Float64.(value.(linking_variables(env.master)))
    t = Float64.(value.(auxiliary_variables(env.master)))

    return Candidate(
        id,
        Dict(:x => x, :t => t),
        Float64(objective_value(model)),
    )
end

function dispatch_candidate!(
    env::MPIAsynchronousBendersEnv,
    worker::Integer,
    candidate::Candidate,
)
    MPI.send(
        (command = :separate, candidate = candidate),
        env.comm;
        dest = worker,
        tag = TAG_WORK,
    )
end

function shutdown_workers!(
    env::MPIAsynchronousBendersEnv,
    busy_workers::Set{Int},
)
    # Every busy worker has exactly one outstanding result when solve! exits.
    # Receive those results before stopping the worker service loops.
    for _ in busy_workers
        status = MPI.Probe(
            env.comm,
            MPI.Status;
            source = MPI.ANY_SOURCE,
            tag = TAG_RESULT,
        )

        MPI.recv(
            env.comm;
            source = status.source,
            tag = TAG_RESULT,
        )
    end

    for worker in env.workers
        MPI.send(
            (command = :stop,),
            env.comm;
            dest = worker,
            tag = TAG_STOP,
        )
    end
end

# -----------------------------------------------------------------------------
# Asynchronous solution procedure
# -----------------------------------------------------------------------------

"""
    solve!(env::MPIAsynchronousBendersEnv)

Run asynchronous Benders coordination on the root rank.

Worker results are processed as they arrive. Cuts are valid even when they
were generated for an older candidate and are therefore incorporated
immediately. A candidate's master objective becomes a lower bound once all
workers have evaluated that candidate. A fully evaluated candidate that is in
`L` for every worker certifies optimality.
"""
function solve!(env::MPIAsynchronousBendersEnv)
    log = Log()
    param = env.param

    MPI.Comm_rank(env.comm) == ROOT ||
        error("MPIAsynchronousBendersEnv.solve! must run on rank 0.")

    workers = env.workers

    next_id = 1
    latest_candidate = master_candidate(env, next_id)
    log.LB = max(log.LB, latest_candidate.master_value)
    log.states[latest_candidate.id] =
        CandidateState(latest_candidate)
    next_id += 1

    for worker in workers
        dispatch_candidate!(env, worker, latest_candidate)
        push!(log.busy_workers, worker)
    end

    try
        while true
            get_sec_remaining(log, param) <= 0.0 && throw(TimeLimitException("MPIAsynchronousBendersEnv: Time limit reached."))

            # receive a result
            status = MPI.Probe(
                env.comm,
                MPI.Status;
                source = MPI.ANY_SOURCE,
                tag = TAG_RESULT,
            )
            worker = Int(status.source)

            message = MPI.recv(
                env.comm;
                source = worker,
                tag = TAG_RESULT,
            )

            message.command === :result || throw(
                ArgumentError("Unexpected MPI message from rank $worker."),
            )

            delete!(log.busy_workers, worker)

            candidate_id = message.candidate_id
            result = message.result

            # Add only cuts violated by the latest candidate. Cuts generated for older
            # candidates remain valid, but may no longer be violated by the latest one.
            violated_cuts = [
                h for h in result.cuts
                if evaluate_violation(
                    h,
                    latest_candidate.values[:x],
                    latest_candidate.values[:t],
                ) > 0.0
            ]
            
            if !isempty(violated_cuts)
                add_cuts!(env.master, violated_cuts)
            
                # Adding a violated cut changes the master and therefore produces a new
                # candidate and a new valid lower bound.
                latest_candidate = master_candidate(env, next_id)
                log.LB = max(log.LB, latest_candidate.master_value)
                log.states[latest_candidate.id] = CandidateState(latest_candidate)
                next_id += 1
            end

            # Record the result for the candidate that the worker actually
            # evaluated. Results for older candidates remain useful for upper-bound updates.
            state = log.states[candidate_id]
            push!(state.completed_workers, worker)
            state.is_in_L[worker] = result.is_in_L

            local_auxiliary_indices = vcat(result.auxiliary_indices...)
            length(local_auxiliary_indices) == length(result.sub_obj_vals) ||
                throw(DimensionMismatch(
                    "MPI result from rank $worker returned " *
                    "$(length(result.sub_obj_vals)) objective values for " *
                    "$(length(local_auxiliary_indices)) auxiliary variables.",
                ))

            state.sub_obj_vals[local_auxiliary_indices] = result.sub_obj_vals

            # Complete evaluation provides the full auxiliary values needed to update
            # the upper bound and optimality gap.
            if fully_evaluated(state, workers)
                if all(isfinite, state.sub_obj_vals)
                    update_upper_bound_and_gap!(
                            state,
                            log,
                            (auxiliary_values, linking_values) ->
                                evaluate_objective(
                                    env.master,
                                    state.candidate.values[:x],
                                    state.sub_obj_vals,
                                ),
                        )
                end
            end

            print_iteration_info(state, log; prefix="")

            if fully_evaluated(state, workers)
                (all(values(state.is_in_L)) || log.gap <= param.gap_tolerance) && break
            end

            # This worker is available again. Give it the latest candidate, which may be
            # newer than the candidate that produced the result just received.
            dispatch_candidate!(env, worker, latest_candidate)
            push!(log.busy_workers, worker)
        end
        env.termination_status = BendersX.Optimal()
        env.obj_value = log.UB
    finally
        shutdown_workers!(env, log.busy_workers)
    end

    return nothing
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
            candidate = message.candidate

            local_is_in_L, local_cuts, local_sub_obj_vals = generate_cuts(
                local_oracle,
                candidate.values[:x],
                candidate.values[:t],
            )

            result = (
                subproblem_indices =
                    vcat(local_oracle.subproblem_indices...),
                auxiliary_indices =
                    copy(local_oracle.auxiliary_indices),
                is_in_L = local_is_in_L,
                cuts = local_cuts,
                sub_obj_vals = local_sub_obj_vals,
            )

            MPI.send(
                (
                    command = :result,
                    candidate_id = candidate.id,
                    result = result,
                ),
                comm;
                dest = root,
                tag = TAG_RESULT,
            )
        else
            throw(ArgumentError("Unknown MPI worker command: $command"))
        end
    end
end

# -----------------------------------------------------------------------------
# Root / worker roles
# -----------------------------------------------------------------------------

if rank == ROOT
    master = Master(
        data;
        optimizer = optimizer_with_attributes(
            optimizer,
            MOI.Silent() => true,
        ),
    )

    env = MPIAsynchronousBendersEnv(
        master,
        COMM,
        collect(1:(nranks - 1)),
        worker_partitions,
        Param(),
        BendersX.NotSolved(),
        Inf,
    )
    
    solve!(env)
    println("rank 0: termination = $(env.termination_status)")
    println("rank 0: objective   = $(env.obj_value)")
else
    serve_worker!(local_oracle, COMM; root = ROOT)
    MPI.Barrier(COMM)
end

if rank == ROOT
    MPI.Barrier(COMM)
end

MPI.Finalize()
