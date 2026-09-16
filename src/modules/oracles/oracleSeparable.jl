"""
    (::Type{T})(
        data,
        master::AbstractMaster;
        model = update_sub_model!,
        scen_idx::Int,
        param::AbstractOracleParam,
        optimizer = DEFAULT_OPTIMIZER,
    ) where T <: AbstractOracle

Scenario-specific constructor interface used by the homogeneous convenience constructor of [`SeparableOracle`](@ref).

When calling

    SeparableOracle(data, master, T, N; ...)

`SeparableOracle` constructs one oracle of type `T` for each subproblem selected by `indices` (all `1:N` by default). To support this form of construction, `T` must implement the constructor above. For global subproblem `j`, `SeparableOracle` calls the constructor with `scen_idx = j`.

This constructor is required only for automatic homogeneous construction. It is not part of the general [`AbstractOracle`](@ref) interface. Oracles that do not implement this constructor can still be used with `SeparableOracle` by constructing them explicitly and passing them to

    SeparableOracle(master, oracles)

# Keywords

- `model`: JuMP modeling function used to construct the subproblem.
- `scen_idx`: Index of the scenario or independent subproblem represented by
  the constructed oracle.
- `param`: Parameter object for the constructed oracle.
- `optimizer`: Optimizer used by the constructed oracle.

# Throws

Throws an [`UnimplementedInterfaceException`](@ref) when a subtype `T` does not implement this constructor and is used with the homogeneous `SeparableOracle(data, master, T, N; ...)` constructor.
"""
(::Type{T})(data, master::AbstractMaster;
            model = update_sub_model!,
            scen_idx::Int,
            param::AbstractOracleParam,
            optimizer = DEFAULT_OPTIMIZER) where T <: AbstractOracle =
    throw(UnimplementedInterfaceException(
        """
        SeparableOracle: 
        Oracle subtype $(T) does not implement the required constructor needed by `SeparableOracle`.

        Expected constructor signature:

          $(T)(data, master::AbstractMaster;
              model = update_sub_model!, scen_idx::Int, param::AbstractOracleParam,
              optimizer = ...)

        Define this constructor for $(T) in order to use it with `SeparableOracle`.
        """
    ))

"""
    SeparableOracleParam <: AbstractOracleParam

Parameters controlling [`SeparableOracle`](@ref).

This parameter container is currently empty and serves as an extension point for future controls related to separable-subproblem evaluation.

Parameters of the individual sub-oracles are stored by those oracles rather than by `SeparableOracle`.

See also: [`SeparableOracle`](@ref), [`AbstractOracleParam`](@ref)
"""
mutable struct SeparableOracleParam <: AbstractOracleParam
    # may contain parameters for scenario handling.
end

"""
    SeparableOracle <: AbstractOracle

Composite oracle for problems with multiple independent subproblems.

`SeparableOracle` composes separation across a collection of independent
subproblems. An instance may represent all subproblems or a subset of them.
Each contained sub-oracle evaluates the common linking-variable candidate and
its corresponding coordinates of the global auxiliary vector. Generated cuts are
copied and embedded in the full auxiliary space.

Sub-oracles may have different concrete types, parameter objects, and auxiliary-variable dimensions.

# Fields
param::SeparableOracleParam: Parameters controlling separable evaluation.
oracles::Vector{AbstractOracle}: Configured oracles for the represented subproblems.
indices::Vector{Vector{Int}}: Global subproblem indices represented by each
component oracle.
auxiliary_indices::Vector{Vector{Int}}: Ordered global auxiliary-variable
positions corresponding to each local sub-oracle's auxiliary coordinates.
dim_auxiliary::Int: Total auxiliary-variable dimension represented by the contained sub-oracles.
dim_global_auxiliary::Int: Dimension of the full global auxiliary space of the master.

# Constructors

    SeparableOracle(
        master::AbstractMaster,
        oracles::AbstractVector{<:AbstractOracle};
        indices = 1:length(oracles),
        auxiliary_indices = nothing,
        param = SeparableOracleParam(),
    )

Construct a `SeparableOracle` from already configured component oracles. Each
entry of `indices` lists the subproblems represented by the corresponding
component. A vector of integers remains shorthand for one subproblem per
component. When `auxiliary_indices` is omitted, the supplied oracles must
represent the full auxiliary space and their positions are inferred consecutively.

    SeparableOracle(
        data,
        master::AbstractMaster,
        oracle_type::Type{T},
        N::Int;
        indices = 1:N,
        auxiliary_indices = nothing,
        model = update_sub_model!,
        sub_oracle_param = BasicOracleParam(),
        param = SeparableOracleParam(),
        optimizer = DEFAULT_OPTIMIZER,
    ) where T <: AbstractOracle

Homogeneous convenience constructor. By default, it constructs one oracle for
each of the `N` subproblems. A subset can be constructed by specifying global
`indices`. For a subset, `auxiliary_indices` must give the ordered positions
in the full master auxiliary space. The same `sub_oracle_param` and `model`
function are passed to every constructed sub-oracle.

For a full `SeparableOracle`, the sum of the sub-oracles'
[`auxiliary_dimension`](@ref) values must equal `master.dim_t`. For a
partitioned instance, each supplied global auxiliary-index vector must have
the same length as the corresponding sub-oracle's auxiliary dimension.

# Throws

Throws a `DimensionMismatch` or `ArgumentError` when the subproblem indices or
auxiliary indices are inconsistent with the supplied sub-oracles or the master
auxiliary space.

See also: [`AbstractOracle`](@ref), [`generate_cuts`](@ref)
"""
mutable struct SeparableOracle <: AbstractOracle
    param::SeparableOracleParam 

    oracles::Vector{AbstractOracle}
    indices::Vector{Vector{Int}}
    auxiliary_indices::Vector{Vector{Int}}
    dim_auxiliary::Int
    dim_global_auxiliary::Int

    function SeparableOracle(
        master::AbstractMaster,
        oracles::AbstractVector{<:AbstractOracle};
        indices = collect(1:length(oracles)),
        auxiliary_indices = nothing,
        param::SeparableOracleParam = SeparableOracleParam(),
    )
        n_local = length(oracles)

        # Grouping preserves which subproblems belong to each component while
        # retaining the original flat-vector constructor as shorthand.
        if all(index -> index isa Integer, indices)
            index_groups = [[Int(index)] for index in indices]
        elseif all(group -> group isa AbstractVector &&
                    all(index -> index isa Integer, group), indices)
            index_groups = [Int.(group) for group in indices]
        else
            throw(ArgumentError(
                "SeparableOracle: indices must contain integers or vectors of integers.",
            ))
        end

        # check the validity of the supplied indices
        length(index_groups) == n_local || throw(DimensionMismatch(
            "SeparableOracle: number of index groups ($(length(index_groups))) must " *
            "equal number of component oracles ($n_local).",
        ))
        all(group -> !isempty(group), index_groups) || throw(ArgumentError(
            "SeparableOracle: every component must represent at least one subproblem.",
        ))
        flat_indices = reduce(vcat, index_groups; init = Int[])
        length(unique(flat_indices)) == length(flat_indices) || throw(ArgumentError(
            "SeparableOracle: subproblem indices must be unique.",
        ))
        all(>(0), flat_indices) || throw(ArgumentError(
            "SeparableOracle: subproblem indices must be positive.",
        ))

        # check the validity of the supplied auxiliary indices
        child_dimensions = auxiliary_dimension.(oracles)
        all(>(0), child_dimensions) || throw(ArgumentError(
            "SeparableOracle: every sub-oracle must have positive auxiliary dimension.",
        ))
        dim_auxiliary = sum(child_dimensions)
        if auxiliary_indices === nothing
            # Without an explicit global mapping, the supplied children must
            # represent the complete auxiliary space in consecutive order.
            dim_auxiliary == master.dim_t || throw(DimensionMismatch(
                "SeparableOracle: supplied sub-oracles represent $dim_auxiliary auxiliary " *
                "variables, but master.dim_t is $(master.dim_t). For a partitioned " *
                "SeparableOracle, provide the corresponding global auxiliary_indices.",
            ))
            mappings = Vector{Vector{Int}}()
            next_index = 1
            for dimension in child_dimensions
                push!(mappings, collect(next_index:(next_index + dimension - 1)))
                next_index += dimension
            end
        else
            length(auxiliary_indices) == n_local || throw(DimensionMismatch(
                "SeparableOracle: number of auxiliary-index mappings " *
                "($(length(auxiliary_indices))) must equal number of component " *
                "oracles ($n_local).",
            ))
            all(
                mapping -> mapping isa AbstractVector &&
                           all(index -> index isa Integer, mapping),
                auxiliary_indices,
            ) || throw(ArgumentError(
                "SeparableOracle: auxiliary_indices must contain vectors of integers.",
            ))
            mappings = [Int.(mapping) for mapping in auxiliary_indices]
            for k in eachindex(mappings)
                length(mappings[k]) == child_dimensions[k] || throw(DimensionMismatch(
                    "SeparableOracle: component $(index_groups[k]) has auxiliary " *
                    "dimension $(child_dimensions[k]), but its global auxiliary " *
                    "indices $(mappings[k]) have length $(length(mappings[k])).",
                ))
                all(index -> 1 <= index <= master.dim_t, mappings[k]) || throw(ArgumentError(
                    "SeparableOracle: auxiliary indices $(mappings[k]) lie outside " *
                    "1:$(master.dim_t).",
                ))
            end
            occupied = reduce(vcat, mappings; init = Int[])
            length(unique(occupied)) == length(occupied) || throw(ArgumentError(
                "SeparableOracle: auxiliary indices must be unique across components.",
            ))
        end

        @info "SeparableOracle: $(length(flat_indices)) assigned subproblems, " *
              "dim_auxiliary=$(dim_auxiliary), " *
              "dim_global_auxiliary=$(master.dim_t), " *
              "$(Threads.nthreads()) threads available for parallel execution"

        new(
            param,
            AbstractOracle[oracles...],
            index_groups,
            mappings,
            dim_auxiliary,
            master.dim_t,
        )
    end
end

function SeparableOracle(
        data,
        master::AbstractMaster,
        oracle_type::Type{T},
        N::Int;
        indices::AbstractVector{<:Integer} = collect(1:N),
        auxiliary_indices = nothing,
        model = update_sub_model!,
        sub_oracle_param::AbstractOracleParam = BasicOracleParam(),
        param::SeparableOracleParam = SeparableOracleParam(),
        optimizer = DEFAULT_OPTIMIZER,
    ) where {T <: AbstractOracle}
        indices = Int.(indices)
        length(unique(indices)) == length(indices) || throw(ArgumentError(
            "SeparableOracle: subproblem indices must be unique.",
        ))
        N == length(indices) || throw(DimensionMismatch(
            "SeparableOracle: number of indices ($(length(indices))) must equal " *
            "number of sub-oracles ($N).",
        ))

        oracles = [
            oracle_type(
                data,
                master;
                model = model,
                scen_idx = j,
                param = deepcopy(sub_oracle_param),
                optimizer = optimizer,
            )
            for j in indices
        ]

        return SeparableOracle(
            master,
            oracles;
            indices = indices,
            auxiliary_indices = auxiliary_indices,
            param = param,
        )
end

is_typical_oracle(oracle::SeparableOracle) =
    all(is_typical_oracle, oracle.oracles)

auxiliary_dimension(oracle::SeparableOracle) = oracle.dim_auxiliary

function embed_local_cut(
    h::Hyperplane,
    child_indices::Vector{Int},
    auxiliary_indices::Vector{Int},
    dim_global_auxiliary::Int,
    dim_x::Int,
)
    length(h.a_x) == dim_x || throw(
        DimensionMismatch(
            "SeparableOracle child $child_indices returned a cut with " *
            "a_x length $(length(h.a_x)); expected $dim_x.",
        ),
    )
    child_dimension = length(auxiliary_indices)
    length(h.a_t) == child_dimension || throw(
        DimensionMismatch(
            "SeparableOracle child $child_indices returned a cut with " *
            "a_t length $(length(h.a_t)); expected $child_dimension.",
        ),
    )

    embedded = Hyperplane(dim_x, dim_global_auxiliary)
    embedded.a_x = copy(h.a_x)
    local_indices, local_values = findnz(h.a_t)
    for k in eachindex(local_indices)
        embedded.a_t[auxiliary_indices[local_indices[k]]] = local_values[k]
    end
    embedded.a_0 = h.a_0
    return embedded
end

"""
    generate_cuts(
        oracle::SeparableOracle,
        x_value::Vector{Float64},
        t_value::Vector{Float64};
        tol_normalize = 1.0,
        time_limit = 3600.0,
    )

Generate Benders cuts by evaluating the sub-oracles represented by this
`SeparableOracle` in parallel.

Each component oracle is evaluated at the common candidate `x_value`. Leaf
oracles receive their corresponding coordinates of `t_value`, while nested
`SeparableOracle`s and `SplitOracle`s receive the full vector because they own
their local-to-global mappings. The returned `sub_obj_vals` follow the order of
the flattened `oracle.indices`.

Returns `(is_in_L, hyperplanes, sub_obj_vals)`, where `is_in_L` indicates whether the candidate is reported as belonging to the feasible regions of all sub-oracles.
"""
function generate_cuts(oracle::SeparableOracle, x_value::Vector{Float64}, t_value::Vector{Float64}; tol_normalize = 1.0, time_limit = 3600.0)
    tic = time()
    n_local = length(oracle.oracles)
    dim_auxiliary = auxiliary_dimension(oracle)
    dim_global_auxiliary = oracle.dim_global_auxiliary

    length(t_value) == dim_global_auxiliary || throw(DimensionMismatch(
        "SeparableOracle received t_value with length $(length(t_value)); " *
        "expected $dim_global_auxiliary.",
    ))

    local_is_in_L = Vector{Bool}(undef, n_local)
    sub_obj_vals = Vector{Vector{Float64}}(undef, n_local)
    hyperplanes = Vector{Vector{Hyperplane}}(undef, n_local)

    try
        Threads.@threads for k=1:n_local
            child = oracle.oracles[k]
            represented_indices = oracle.indices[k]
            child_auxiliary_indices = oracle.auxiliary_indices[k]
            child_dimension = length(child_auxiliary_indices)

            # Composite children own a local-to-global mapping, so they receive
            # the global candidate and already return globally embedded cuts.
            child_uses_global_space = child isa SeparableOracle || child isa SplitOracle
            child_t_value = child_uses_global_space ? t_value : t_value[child_auxiliary_indices]

            child_is_in_L, child_hyperplanes, child_obj_vals = generate_cuts(
                child,
                x_value,
                child_t_value;
                tol_normalize = tol_normalize,
                time_limit = get_sec_remaining(tic, time_limit),
            )
            length(child_obj_vals) == child_dimension || throw(
                DimensionMismatch(
                    "SeparableOracle component for subproblems $represented_indices returned " *
                    "$(length(child_obj_vals)) objective values; expected $child_dimension.",
                ),
            )

            local_is_in_L[k] = child_is_in_L
            sub_obj_vals[k] = child_obj_vals
            if child_uses_global_space
                hyperplanes[k] = child_hyperplanes
            else
                hyperplanes[k] = [
                    embed_local_cut(
                        h,
                        represented_indices,
                        child_auxiliary_indices,
                        dim_global_auxiliary,
                        length(x_value),
                    ) for h in child_hyperplanes
                ]
            end
        end
    catch err
        err isa CompositeException || rethrow()
        task_failure = first(err.exceptions)
        task_failure isa TaskFailedException || rethrow()
        failures = current_exceptions(task_failure.task; backtrace = false)
        isempty(failures) && rethrow()
        throw(first(failures).exception)
    end

    cuts = reduce(vcat, hyperplanes; init = Hyperplane[])
    values = reduce(vcat, sub_obj_vals; init = Float64[])
    is_in_L = all(local_is_in_L)

    if is_in_L
        cuts = [Hyperplane(length(x_value), dim_global_auxiliary)]
    end

    return is_in_L, cuts, values
end
