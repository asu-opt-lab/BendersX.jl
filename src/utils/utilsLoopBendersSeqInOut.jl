"""
    BendersSeqInOutParam <: AbstractBendersSeqParam

Parameters controlling [`BendersSeqInOut`](@ref).

In addition to the standard sequential Benders stopping criteria, these parameters control the in-out stabilization procedure.

# Fields

- `time_limit::Float64`: Maximum wall-clock time allowed for the algorithm, in seconds.
- `gap_tolerance::Float64`: Relative gap tolerance for termination.
- `halt_limit::Int`: Maximum number of consecutive iterations without sufficient lower-bound improvement.
- `iter_limit::Int`: Maximum number of Benders iterations.
- `verbose::Bool`: Whether to print iteration information.
- `stabilizing_x::Vector{Float64}`: Initial stabilizing point in the first-stage variable space.
- `α::Float64`: Weight used to update the stabilizing point.
- `λ::Float64`: Weight used to form the perturbed oracle query point.

`stabilizing_x` must have the same dimension as the master's first-stage variable vector `x`.

See also: [`BendersSeqInOut`](@ref), [`BendersSeqParam`](@ref)
"""
mutable struct BendersSeqInOutParam <: AbstractBendersSeqParam

    time_limit::Float64
    gap_tolerance::Float64
    halt_limit::Int
    iter_limit::Int
    verbose::Bool

    stabilizing_x::Vector{Float64} 
    α::Float64
    λ::Float64

    function BendersSeqInOutParam(; 
                        time_limit::Float64 = 7200.0, 
                        gap_tolerance::Float64 = 1e-6, 
                        halt_limit::Int = 10000, 
                        iter_limit::Int = 1000000, 
                        verbose::Bool = true,
                        stabilizing_x::Vector{Float64}, # must be provided
                        α::Float64 = 0.9,
                        λ::Float64 = 0.1
                        ) 
        
        new(time_limit, gap_tolerance, halt_limit, iter_limit, verbose, stabilizing_x, α, λ)
    end
end

"""
    is_terminated(
        state::BendersSeqState,
        log::BendersSeqLog,
        param::BendersSeqInOutParam,
    )

Check whether the sequential Benders In-Out loop should terminate.

The loop terminates when the current candidate solution is in the oracle feasible region or when the current optimality gap is within `param.gap_tolerance`.

Time-limit handling is performed by the surrounding solve loop.

Returns `true` if the loop should terminate and `false` otherwise.
"""
function is_terminated(state::BendersSeqState, log::BendersSeqLog, param::BendersSeqInOutParam)
    return state.is_in_L || state.gap <= param.gap_tolerance
end