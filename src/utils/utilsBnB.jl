"""
    AbstractBendersBnBParam

Abstract supertype for parameter containers used by Benders branch-and-bound environments.

See also: [`BendersBnBParam`](@ref), [`AbstractLoopParam`](@ref)
"""
abstract type AbstractBendersBnBParam end

"""
    AbstractBendersBnBState

Abstract supertype for state information associated with a node processed during Benders branch-and-bound.

A concrete B&B state records the candidate solution, oracle evaluation, and cut-generation information associated with a processed node.

See also: `BendersBnBState`, [`AbstractLoopState`](@ref)
"""
abstract type AbstractBendersBnBState end

"""
    AbstractBendersBnBLog

Abstract supertype for logs that record the progress of a Benders branch-and-bound environment.

Unlike sequential Benders logs, a B&B log records information associated with processed nodes rather than iterations of a cutting-plane loop.

See also: `BendersBnBLog`, [`AbstractLoopLog`](@ref)
"""
abstract type AbstractBendersBnBLog end


"""
    BendersBnBState <: AbstractBendersBnBState

State information associated with a node processed during Benders branch-and-bound.

# Fields

- `oracle_time::Float64`: Time spent evaluating the oracle at the node.
- `values::Dict{Symbol,Vector{Float64}}`: Candidate solution values at the node, including `:x` and `:t`.
- `f_x::Vector{Float64}`: Subproblem objective values evaluated at the candidate solution. Entries may be `Inf` when no finite objective value is available.
- `is_in_L::Bool`: Whether the candidate solution belongs to the oracle's feasible region.
- `node::Int`: Identifier of the processed branch-and-bound node.
- `num_cuts::Int`: Number of cuts generated at the node.

See also: [`BendersBnBLog`](@ref), [`BendersBnB`](@ref)
"""
mutable struct BendersBnBState <: AbstractBendersBnBState
    oracle_time::Float64
    values::Dict{Symbol,Vector{Float64}}
    f_x::Vector{Float64}
    is_in_L::Bool
    node::Int
    num_cuts::Int
    function BendersBnBState()
        new(0.0, Dict(:x => Vector{Float64}(), :t => Vector{Float64}()), Vector{Float64}(), false, 0, 0)
    end
end


"""
    BendersBnBLog <: AbstractBendersBnBLog

Log of node-level activity during a Benders branch-and-bound solve.

# Fields

- `nodes::Vector{BendersBnBState}`: States recorded for processed branch-and-bound nodes.
- `n_enter_nodes::Int`: Number of nodes processed by the callbacks.
- `n_lazy_cuts::Int`: Total number of lazy Benders cuts generated.
- `n_user_cuts::Int`: Total number of user Benders cuts generated.
- `start_time::Float64`: Time at which the solve started.
- `preprocessing_time::Float64`: Time spent in preprocessing before branch-and-bound.
- `total_time::Float64`: Total elapsed time of the solve.
- `fractional_nodes_since_cut::Int`: Number of processed fractional nodes since the last user-cut evaluation.

See also: [`BendersBnBState`](@ref), [`BendersBnBParam`](@ref), [`BendersBnB`](@ref)
"""
mutable struct BendersBnBLog <: AbstractBendersBnBLog
    nodes::Vector{BendersBnBState}
    n_enter_nodes::Int
    n_lazy_cuts::Int
    n_user_cuts::Int
    start_time::Float64
    preprocessing_time::Float64
    total_time::Float64
    fractional_nodes_since_cut::Int
    function BendersBnBLog()
        new(Vector{BendersBnBState}(), 0, 0, 0, time(), 0.0, 0.0, 0)
    end
end

"""
    BendersBnBParam <: AbstractBendersBnBParam

Parameters controlling [`BendersBnB`](@ref).

# Fields

- `time_limit::Float64`: Maximum wall-clock time allowed for the algorithm, in seconds.
- `gap_tolerance::Float64`: Relative optimality-gap tolerance passed to the MIP solver.
- `verbose::Bool`: Whether to print solve and callback information.

# Constructor

    BendersBnBParam(;
        time_limit = 7200.0,
        gap_tolerance = 1e-6,
        verbose = true,
    )

Construct branch-and-bound Benders parameters with the specified global time limit, solver gap tolerance, and verbosity.

See also: [`BendersBnB`](@ref), [`AbstractBendersBnBParam`](@ref)
"""
mutable struct BendersBnBParam <: AbstractBendersBnBParam
    time_limit::Float64
    gap_tolerance::Float64
    verbose::Bool

    function BendersBnBParam(; 
                        time_limit::Float64 = 7200.0, 
                        gap_tolerance::Float64 = 1e-6, 
                        verbose::Bool = true
                        ) 
        new(time_limit, gap_tolerance, verbose)
    end
end 

"""
    record_node!(
        log::BendersBnBLog,
        state::BendersBnBState,
        is_lazy_cut::Bool,
    )

Record a processed branch-and-bound node in the B&B log.

The node state is appended to `log.nodes`, the processed-node counter is incremented, and the corresponding lazy- or user-cut counter is updated using `state.num_cuts`.

# Arguments

- `log::BendersBnBLog`: B&B log to update.
- `state::BendersBnBState`: State of the processed node.
- `is_lazy_cut::Bool`: Whether the generated cuts are lazy cuts.

Returns `nothing`.
"""
function record_node!(log::BendersBnBLog, state::BendersBnBState, is_lazy_cut::Bool)
    push!(log.nodes, state)
    log.n_enter_nodes += 1
    log.n_lazy_cuts += is_lazy_cut ? state.num_cuts : 0
    log.n_user_cuts += !is_lazy_cut ? state.num_cuts : 0
end

"""
    get_sec_remaining(
        log::BendersBnBLog,
        param::BendersBnBParam,
    )

Return the number of seconds remaining in the Benders branch-and-bound time budget.

The remaining time is computed from `log.start_time` and `param.time_limit`, and is clamped to zero.

See also: [`get_sec_remaining`](@ref)
"""
function get_sec_remaining(log::BendersBnBLog, param::BendersBnBParam)
    return get_sec_remaining(log.start_time, param.time_limit)
end

"""
    to_dataframe(
        env::AbstractBendersBnB,
        log::BendersBnBLog,
    ) -> DataFrame

Convert the Benders branch-and-bound log and solve result into a one-row summary `DataFrame`.

The returned data frame contains solver-level node statistics, preprocessing time, total solve time, objective bound, objective value, relative optimality gap, and lazy/user-cut counts.

If the master optimizer has not been called, the objective and bound fields are reported as unavailable.

See also: [`BendersBnB`](@ref)
"""
function to_dataframe(env::AbstractBendersBnB, log::BendersBnBLog)
    model = master_model(env.master)
    if termination_status(model) == MOI.OPTIMIZE_NOT_CALLED
        return DataFrame(
            node_count = 0,
            preprocessing_time = log.preprocessing_time,
            time = log.total_time,
            obj_bound = -Inf,
            obj_val = Inf,
            rel_gap = Inf,
            n_lazy_cuts = log.n_lazy_cuts,
            n_user_cuts = log.n_user_cuts
        )
    else
        return DataFrame(
            node_count = JuMP.node_count(model),
            preprocessing_time = log.preprocessing_time,
            time = log.total_time,
            obj_bound = JuMP.objective_bound(model),
            obj_val = env.obj_value,
            rel_gap = has_values(model) ? JuMP.relative_gap(model) : Inf,
            n_lazy_cuts = log.n_lazy_cuts,
            n_user_cuts = log.n_user_cuts
        )
    end
end
