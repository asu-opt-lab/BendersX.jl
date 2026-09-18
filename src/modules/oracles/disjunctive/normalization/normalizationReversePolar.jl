# -----------------------------------------------------------------------------
# Reverse-polar normalization
# -----------------------------------------------------------------------------

"""
    ReversePolarNormalization <: AbstractNormalization

Reverse-polar normalization for DCGLP-based disjunctive cut generation.

The normalization direction can be determined from a core point and the current candidate solution, or specified as a fixed direction.

# Fields

- `core_point_x::Union{Nothing,Vector{Float64}}`: `x` component of the core point used to determine the normalization direction.
- `core_point_t::Union{Nothing,Vector{Float64}}`: `t` component of the core point used to determine the normalization direction.
- `core_direction_x::Union{Nothing,Vector{Float64}}`: `x` component of a fixed normalization direction.
- `core_direction_t::Union{Nothing,Vector{Float64}}`: `t` component of a fixed normalization direction.
- `use_core_point::Bool`: Whether the normalization direction is determined from a core point.

# Constructor

    ReversePolarNormalization(;
        core_point_x = nothing,
        core_point_t = nothing,
        core_direction_x = nothing,
        core_direction_t = nothing,
    )

Construct a reverse-polar normalization using either a core point or a fixed normalization direction.

A core point and a fixed direction cannot be specified simultaneously. When a core point or direction is provided, both its `x` and `t` components must be specified and must have the same dimensions as the corresponding oracle variables.

When neither a core point nor a fixed direction is provided, a vertical direction with zeros in the `x` space and ones in the oracle's auxiliary-variable space is used by default. The default vertical direction is initialized when the normalization is prepared for a specific split oracle. Its validity is problem-dependent and is not guaranteed for all master formulations.

See also: [`AbstractNormalization`](@ref), [`LpDistanceNormalization`](@ref), [`SplitOracle`](@ref)
"""
mutable struct ReversePolarNormalization <: AbstractNormalization
    core_point_x::Union{Nothing, Vector{Float64}}
    core_point_t::Union{Nothing, Vector{Float64}}
    core_direction_x::Union{Nothing, Vector{Float64}}
    core_direction_t::Union{Nothing, Vector{Float64}}
    use_core_point::Bool

    function ReversePolarNormalization(;
        core_point_x::Union{Nothing, Vector{Float64}} = nothing,
        core_point_t::Union{Nothing, Vector{Float64}} = nothing,
        core_direction_x::Union{Nothing, Vector{Float64}} = nothing,
        core_direction_t::Union{Nothing, Vector{Float64}} = nothing,
    )

        has_point = core_point_x !== nothing || core_point_t !== nothing
        has_direction = core_direction_x !== nothing || core_direction_t !== nothing
        use_core_point = has_point

        if !has_point && !has_direction
            @warn "ReversePolarNormalization: No core point or direction provided; " *
            "using the default vertical reverse-polar direction. Its validity " *
            "is not guaranteed for all master formulations."
        end

        has_point && has_direction &&
            throw(ArgumentError("ReversePolarNormalization: Provide either `core_point_x`/`core_point_t` or `core_direction_x`/`core_direction_t`, not both."))

        if has_point
            core_point_x !== nothing ||
                throw(ArgumentError("ReversePolarNormalization: `core_point_x` must be provided when `core_point_t` is provided."))
            core_point_t !== nothing ||
                throw(ArgumentError("ReversePolarNormalization: `core_point_t` must be provided when `core_point_x` is provided."))
            isempty(core_point_x) && throw(ArgumentError("ReversePolarNormalization: `core_point_x` must be non-empty."))
            isempty(core_point_t) && throw(ArgumentError("ReversePolarNormalization: `core_point_t` must be non-empty."))
        end

        if has_direction
            core_direction_x !== nothing ||
                throw(ArgumentError("ReversePolarNormalization: `core_direction_x` must be provided when `core_direction_t` is provided."))
            core_direction_t !== nothing ||
                throw(ArgumentError("ReversePolarNormalization: `core_direction_t` must be provided when `core_direction_x` is provided."))
            isempty(core_direction_x) && throw(ArgumentError("ReversePolarNormalization: `core_direction_x` must be non-empty."))
            isempty(core_direction_t) && throw(ArgumentError("ReversePolarNormalization: `core_direction_t` must be non-empty."))
            LinearAlgebra.norm(vcat(core_direction_x, core_direction_t), Inf) > 0.0 ||
                throw(ArgumentError("ReversePolarNormalization: `core_direction_x`/`core_direction_t` must not define the zero direction."))
        end

        return new(
            core_point_x,
            core_point_t,
            core_direction_x,
            core_direction_t,
            use_core_point,
        )
    end
end

function add_normalization_constraint!(
    normalization::ReversePolarNormalization,
    master::AbstractMaster,
    dcglp::Model,
    tau::VariableRef,
    sx::AbstractVector{VariableRef},
    st::AbstractVector{VariableRef},
)
    initialize_reverse_polar!(normalization, length(sx), length(st))

    @constraint(dcglp, con_reverse_polar_x[j in eachindex(sx)], sx[j] + 0.0 * tau == 0.0)
    @constraint(dcglp, con_reverse_polar_t[j in eachindex(st)], st[j] + 0.0 * tau == 0.0)
    
    return nothing
end

function initialize_reverse_polar!(
    normalization::ReversePolarNormalization,
    dim_x::Int,
    dim_t::Int,
)
    if normalization.use_core_point
        check_reverse_polar_dimension(normalization.core_point_x, dim_x, "core_point_x")
        check_reverse_polar_dimension(normalization.core_point_t, dim_t, "core_point_t")
        return nothing
    end

    if normalization.core_direction_x === nothing && normalization.core_direction_t === nothing
        normalization.core_direction_x = zeros(dim_x)
        normalization.core_direction_t = ones(dim_t)
        return nothing
    end

    check_reverse_polar_dimension(normalization.core_direction_x, dim_x, "core_direction_x")
    check_reverse_polar_dimension(normalization.core_direction_t, dim_t, "core_direction_t")
    return nothing
end

function check_reverse_polar_dimension(vector::Vector{Float64}, expected::Int, name::String)
    length(vector) == expected ||
        throw(DimensionMismatch("`$name` has length $(length(vector)) but expected $expected."))
    return nothing
end

function is_applicable(
    normalization::ReversePolarNormalization,
    oracle::AbstractDisjunctiveOracle,
    x_value::Vector{Float64},
    t_value::Vector{Float64},
)
    # Check whether the reverse-polar normalization is valid/useful
    # for this candidate.
    normalization.use_core_point || return true

    direction_x = x_value .- normalization.core_point_x
    direction_t = t_value .- normalization.core_point_t

    if LinearAlgebra.norm(vcat(direction_x, direction_t), Inf) <= oracle.param.zero_tol
        @info "ReversePolarNormalization: Candidate solution is too close to the core point; falling back to typical Benders cuts."
        return false
    end

    return true
end

function update_dcglp_for_candidate!(normalization::ReversePolarNormalization, dcglp::Model, x_value::Vector{Float64}, t_value::Vector{Float64})

    set_normalized_rhs.(dcglp[:conx], x_value)
    set_normalized_rhs.(dcglp[:cont], t_value)

    direction_x, direction_t =
        normalization.use_core_point ?
        (x_value .- normalization.core_point_x, t_value .- normalization.core_point_t) :
        (.-normalization.core_direction_x, .-normalization.core_direction_t)
    update_reverse_polar_constraints!(dcglp, direction_x, direction_t)
end

function update_reverse_polar_constraints!(
    dcglp::Model,
    direction_x::Vector{Float64},
    direction_t::Vector{Float64}
)   
    tau = dcglp[:tau]

    for j in eachindex(direction_x)
        set_normalized_coefficient(dcglp[:con_reverse_polar_x][j], tau, direction_x[j])
    end
    for j in eachindex(direction_t)
        set_normalized_coefficient(dcglp[:con_reverse_polar_t][j], tau, direction_t[j])
    end

    return nothing
end

function update_dcglp_upper_bound_and_gap!(
    ::ReversePolarNormalization,
    state::DcglpState,
    log::DcglpLog,
    t_value::Vector{Float64};
    zero_tol::Float64 = 1e-9
)
    # todo: consider computing upper bound based on projection onto half-ray; Currently it's based on `is_in_L` indicator.

    previous_ub = log.n_iter >= 1 ? log.iterations[end].UB : Inf
    state.UB = state.is_in_L[1] && state.is_in_L[2] ? min(previous_ub, state.LB) : previous_ub
    if !isfinite(state.UB)
        state.gap = Inf
    elseif isapprox(state.UB, state.LB; atol = zero_tol)
        state.gap = 0.0
    elseif abs(state.UB) <= zero_tol
        state.gap = Inf
    else
        state.gap = max(
            0.0,
            (state.UB - state.LB) / abs(state.UB),
        )
    end
    
    return nothing
end

function disjunctive_cut_normalization_value(
    ::ReversePolarNormalization,
    dcglp::Model,
    gamma_x::Vector{Float64},
    gamma_t::Vector{Float64}
)
    tau = dcglp[:tau]
    direction_x = [normalized_coefficient(con, tau) for con in dcglp[:con_reverse_polar_x]]
    direction_t = [normalized_coefficient(con, tau) for con in dcglp[:con_reverse_polar_t]]
    direction_value = dot(gamma_x, direction_x) + dot(gamma_t, direction_t)
    !iszero(direction_value) ||
        throw(AlgorithmException("ReversePolarNormalization cut normalization failed because the directional support is numerically zero."))
    return direction_value
end
