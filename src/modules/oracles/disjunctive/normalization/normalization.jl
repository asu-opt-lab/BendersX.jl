"""
    AbstractNormalization

Abstract supertype for normalization schemes used in DCGLP-based disjunctive cut generation.

A normalization defines the normalization constraint added to the DCGLP, the candidate-dependent updates required before solving the DCGLP, the normalization used to evaluate the DCGLP upper bound, and the scaling applied when constructing the resulting disjunctive cut.

Available normalization schemes include:

- [`LpDistanceNormalization`](@ref): Uses an ``L_p``-distance normalization.
- [`ReversePolarNormalization`](@ref): Uses a reverse-polar normalization.

See also: [`SplitOracle`](@ref), [`SplitOracleParam`](@ref)
"""
abstract type AbstractNormalization end

# ---------------------------------------------------------------------------- 
# Common interface 
# ----------------------------------------------------------------------------
"""
    add_normalization_constraint!(
        normalization::AbstractNormalization,
        master::AbstractMaster,
        dcglp::Model,
        tau::VariableRef,
        sx::AbstractVector{VariableRef},
        st::AbstractVector{VariableRef},
    )

Add the normalization-specific constraint to `dcglp`.

The normalization is applied to the common DCGLP variables `tau`, `sx`, and `st`. Concrete normalization types must implement this method.
"""
function add_normalization_constraint!(
    normalization::AbstractNormalization,
    master::AbstractMaster,
    dcglp::Model,
    tau::VariableRef,
    sx::AbstractVector{VariableRef},
    st::AbstractVector{VariableRef},
)
    throw(UnimplementedInterfaceException("update add_normalization_constraint! for $(typeof(normalization))"))
end

is_applicable(
    ::AbstractNormalization,
    ::AbstractDisjunctiveOracle,
    x_value::Vector{Float64},
    t_value::Vector{Float64},
) = true

function update_dcglp_for_candidate!(normalization::AbstractNormalization, dcglp::Model, x_value::Vector{Float64}, t_value::Vector{Float64})
    throw(UnimplementedInterfaceException("update update_dcglp_for_candidate! for $(typeof(normalization))"))
end

function update_dcglp_upper_bound_and_gap!(
    normalization::AbstractNormalization,
    state::DcglpState,
    log::DcglpLog,
    t_value::Vector{Float64},
)
    throw(UnimplementedInterfaceException("update update_dcglp_upper_bound_and_gap! for $(typeof(normalization))"))
end

function disjunctive_cut_normalization_value(
    normalization::AbstractNormalization,
    dcglp::Model,
    gamma_x::Vector{Float64},
    gamma_t::Vector{Float64},
)
    throw(UnimplementedInterfaceException("update disjunctive_cut_normalization_value for $(typeof(normalization))"))
end

include("normalizationLpDistance.jl")
include("normalizationReversePolar.jl")