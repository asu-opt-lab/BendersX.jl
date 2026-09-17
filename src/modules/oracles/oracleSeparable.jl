"""
    (::Type{T})(
        data,
        master::AbstractMaster;
        model = update_sub_model!,
        subproblem_idx::Int,
        param::AbstractOracleParam,
        optimizer = DEFAULT_OPTIMIZER,
    ) where T <: AbstractOracle

Subproblem-specific constructor interface used by the homogeneous convenience constructor of [`SeparableOracle`](@ref).

When calling

    SeparableOracle(data, master, T, N; ...)

`SeparableOracle` constructs one oracle of type `T` for each subproblem selected by
`subproblem_indices` (all `1:N` by default). To support this form of construction,
`T` must implement the constructor above. For global subproblem `j`,
`SeparableOracle` calls the constructor with `subproblem_idx = j`.

This constructor is required only for automatic homogeneous construction. It is not part of the general [`AbstractOracle`](@ref) interface. Oracles that do not implement this constructor can still be used with `SeparableOracle` by constructing them explicitly and passing them to

    SeparableOracle(master, oracles)

# Keywords

- `model`: JuMP modeling function used to construct the subproblem.
- `subproblem_idx`: Index of the independent subproblem represented by
  the constructed oracle.
- `param`: Parameter object for the constructed oracle.
- `optimizer`: Optimizer used by the constructed oracle.

# Throws

Throws an [`UnimplementedInterfaceException`](@ref) when a subtype `T` does not implement this constructor and is used with the homogeneous `SeparableOracle(data, master, T, N; ...)` constructor.
"""
(::Type{T})(data, master::AbstractMaster;
            model = update_sub_model!,
            subproblem_idx::Int,
            param::AbstractOracleParam,
            optimizer = DEFAULT_OPTIMIZER) where T <: AbstractOracle =
    throw(UnimplementedInterfaceException(
        """
        SeparableOracle: 
        Oracle subtype $(T) does not implement the required constructor needed by `SeparableOracle`.

        Expected constructor signature:

          $(T)(data, master::AbstractMaster;
              model = update_sub_model!, subproblem_idx::Int, param::AbstractOracleParam,
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
    # May contain parameters for subproblem handling.
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
subproblem_indices::Vector{Vector{Int}}: Global subproblem indices represented by each
component oracle.
auxiliary_indices::Vector{Vector{Int}}: Global auxiliary-variable
positions corresponding to each local sub-oracle's auxiliary coordinates.
dim_auxiliary::Int: Total auxiliary-variable dimension represented by the contained sub-oracles.
dim_global_auxiliary::Int: Dimension of the full global auxiliary space of the master.

# Constructors

    SeparableOracle(
        master::AbstractMaster,
        oracles::AbstractVector{<:AbstractOracle};
        subproblem_indices = 1:length(oracles),
        auxiliary_indices = nothing,
        param = SeparableOracleParam(),
    )

Construct a `SeparableOracle` from already configured component oracles. Each
entry of `subproblem_indices` lists the subproblems represented by the corresponding
component oracle. A vector of integers remains shorthand for one subproblem per component oracle. When `auxiliary_indices` is omitted, each subproblem must represent one auxiliary variable, and `subproblem_indices` also determine the global auxiliary indices.

    SeparableOracle(
        data,
        master::AbstractMaster,
        oracle_type::Type{T},
        N::Int;
        subproblem_indices = 1:N,
        auxiliary_indices = nothing,
        model = update_sub_model!,
        sub_oracle_param = BasicOracleParam(),
        param = SeparableOracleParam(),
        optimizer = DEFAULT_OPTIMIZER,
    ) where T <: AbstractOracle

Homogeneous convenience constructor. By default, it constructs one oracle for
each of the `N` subproblems. A subset can be constructed by specifying global
`subproblem_indices`. When each subproblem represents one auxiliary
variable, `auxiliary_indices` may be omitted and `subproblem_indices` also
determine the global auxiliary positions. Otherwise, `auxiliary_indices` must
give the positions in the full master auxiliary space. The same
`sub_oracle_param` and `model` function are passed to every constructed
sub-oracle.

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
    subproblem_indices::Vector{Vector{Int}}
    auxiliary_indices::Vector{Vector{Int}}
    dim_auxiliary::Int
    dim_global_auxiliary::Int

    function SeparableOracle(
        master::AbstractMaster,
        oracles::AbstractVector{<:AbstractOracle};
        subproblem_indices = 1:length(oracles),
        auxiliary_indices = nothing,
        param::SeparableOracleParam = SeparableOracleParam(),
    )
        n_local = length(oracles)
        n_local > 0 || throw(ArgumentError(
            "SeparableOracle: at least one component oracle is required.",
        ))
        
        # Allow each component oracle to represent one or more subproblems.
        # A flat vector such as [1, 2, 3] is shorthand for [[1], [2], [3]].
        if all(index -> index isa Integer, subproblem_indices)
            subproblem_groups = [[Int(index)] for index in subproblem_indices]
        elseif all(group -> group isa AbstractVector &&
                    all(index -> index isa Integer, group), subproblem_indices)
            subproblem_groups = [Int.(group) for group in subproblem_indices]
        else
            throw(ArgumentError(
                "SeparableOracle: subproblem_indices must contain integers or vectors of integers.",
            ))
        end

        # check the validity of the supplied subproblem indices
        length(subproblem_groups) == n_local || throw(DimensionMismatch(
            "SeparableOracle: number of subproblem index groups ($(length(subproblem_groups))) must " *
            "equal number of component oracles ($n_local).",
        ))

        all(group -> !isempty(group), subproblem_groups) || throw(ArgumentError(
            "SeparableOracle: each component oracle must represent at least one subproblem.",
        ))

        flat_subproblem_indices = vcat(subproblem_groups...)

        length(unique(flat_subproblem_indices)) == length(flat_subproblem_indices) || throw(
            ArgumentError(
                "SeparableOracle: subproblem indices must be unique across components.",
            ),
        )

        all(>(0), flat_subproblem_indices) || throw(ArgumentError(
            "SeparableOracle: subproblem indices must be positive integers.",
        ))

        # check the validity of the supplied auxiliary indices
        auxiliary_dimensions = auxiliary_dimension.(oracles)

        all(>(0), auxiliary_dimensions) || throw(ArgumentError(
            "SeparableOracle: each component oracle must have a positive " *
            "auxiliary dimension.",
        ))

        dim_auxiliary = sum(auxiliary_dimensions)

        if auxiliary_indices === nothing
            for k in eachindex(subproblem_groups)
                length(subproblem_groups[k]) == auxiliary_dimensions[k] || throw(
                    DimensionMismatch(
                        "SeparableOracle: when auxiliary_indices is omitted, " *
                        "component oracle $k represents " *
                        "$(length(subproblem_groups[k])) subproblems but has " *
                        "auxiliary dimension $(auxiliary_dimensions[k]).",
                    ),
                )
            end

            all(<=(master.dim_t), flat_subproblem_indices) || throw(ArgumentError(
                "SeparableOracle: when auxiliary_indices is omitted, subproblem " *
                "indices also identify global auxiliary variables and must lie " *
                "within 1:$(master.dim_t); received $(flat_subproblem_indices).",
            ))

            mappings = [copy(group) for group in subproblem_groups]
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
                mapping = mappings[k]

                length(mapping) == auxiliary_dimensions[k] || throw(
                    DimensionMismatch(
                        "SeparableOracle: component oracle $k represents " *
                        "$(auxiliary_dimensions[k]) auxiliary variables, but " *
                        "its auxiliary_indices has length $(length(mapping)).",
                    ),
                )

                all(1 <= index <= master.dim_t for index in mapping) || throw(
                    ArgumentError(
                        "SeparableOracle: auxiliary_indices for component oracle " *
                        "$k must lie within 1:$(master.dim_t); received $(mapping).",
                    ),
                )
            end

            occupied = vcat(mappings...)

            length(unique(occupied)) == length(occupied) || throw(
                ArgumentError(
                    "SeparableOracle: auxiliary_indices must be unique across components.",
                ),
            )
        end

        @info "SeparableOracle: $(n_local) component oracles representing " *
              "$(length(flat_subproblem_indices)) subproblems, " *
              "dim_auxiliary=$(dim_auxiliary), " *
              "dim_global_auxiliary=$(master.dim_t), " *
              "$(Threads.nthreads()) threads available for parallel execution"

        new(
            param,
            AbstractOracle[oracles...],
            subproblem_groups,
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
        subproblem_indices::AbstractVector{<:Integer} = 1:N,
        auxiliary_indices = nothing,
        model = update_sub_model!,
        sub_oracle_param::AbstractOracleParam = BasicOracleParam(),
        param::SeparableOracleParam = SeparableOracleParam(),
        optimizer = DEFAULT_OPTIMIZER,
    ) where {T <: AbstractOracle}

        N > 0 || throw(ArgumentError(
            "SeparableOracle: number of subproblems N must be positive.",
        ))
        
        subproblem_indices = Int.(subproblem_indices)

        length(subproblem_indices) == N || throw(DimensionMismatch(
            "SeparableOracle: N ($N) must equal the number of subproblem indices " *
            "($(length(subproblem_indices))).",
        ))
        
        length(unique(subproblem_indices)) == length(subproblem_indices) || throw(ArgumentError(
            "SeparableOracle: subproblem indices must be unique.",
        ))
        
        oracles = [
            oracle_type(
                data,
                master;
                model = model,
                subproblem_idx = j,
                param = deepcopy(sub_oracle_param),
                optimizer = optimizer,
            )
            for j in subproblem_indices
        ]

        return SeparableOracle(
            master,
            oracles;
            subproblem_indices = subproblem_indices,
            auxiliary_indices = auxiliary_indices,
            param = param,
        )
end

is_typical_oracle(oracle::SeparableOracle) =
    all(is_typical_oracle, oracle.oracles)

auxiliary_dimension(oracle::SeparableOracle) = oracle.dim_auxiliary

function embed_local_cut(
    h::Hyperplane,
    subproblem_indices::Vector{Int},
    auxiliary_indices::Vector{Int},
    dim_global_auxiliary::Int,
    dim_x::Int,
)
    length(h.a_x) == dim_x || throw(
        DimensionMismatch(
            "SeparableOracle: component oracle representing subproblems " *
            "$subproblem_indices returned a cut with a_x length " *
            "$(length(h.a_x)); expected $dim_x."
        ),
    )

    auxiliary_dimension = length(auxiliary_indices)
    length(h.a_t) == auxiliary_dimension || throw(
        DimensionMismatch(
            "SeparableOracle: component oracle representing subproblems " *
            "$subproblem_indices returned a cut with a_t length " *
            "$(length(h.a_t)); expected $auxiliary_dimension.",
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
the flattened `oracle.subproblem_indices`.

Returns `(is_in_L, hyperplanes, sub_obj_vals)`, where `is_in_L` indicates whether the candidate is reported as belonging to the feasible regions of all sub-oracles.
"""
function generate_cuts(oracle::SeparableOracle, x_value::Vector{Float64}, t_value::Vector{Float64}; tol_normalize = 1.0, time_limit = 3600.0)
    tic = time()
    n_local = length(oracle.oracles)
    dim_global_auxiliary = oracle.dim_global_auxiliary

    length(t_value) == dim_global_auxiliary || throw(DimensionMismatch(
        "SeparableOracle received t_value with length $(length(t_value)); " *
        "expected $dim_global_auxiliary.",
    ))

    local_is_in_L = Vector{Bool}(undef, n_local)
    sub_obj_vals = Vector{Vector{Float64}}(undef, n_local)
    hyperplanes = Vector{Vector{Hyperplane}}(undef, n_local)

    try
        Threads.@threads for k in eachindex(oracle.oracles)
            component_oracle = oracle.oracles[k]
            subproblem_indices = oracle.subproblem_indices[k]
            auxiliary_indices = oracle.auxiliary_indices[k]
            local_auxiliary_dimension = length(auxiliary_indices)

            # Composite children own a local-to-global mapping, so they receive
            # the global candidate and already return globally embedded cuts.
            uses_global_auxiliary_space = component_oracle isa SeparableOracle || component_oracle isa SplitOracle
            
            component_t_value = uses_global_auxiliary_space ? t_value : t_value[auxiliary_indices]

            component_is_in_L, component_hyperplanes, component_obj_vals = generate_cuts(
                component_oracle,
                x_value,
                component_t_value;
                tol_normalize = tol_normalize,
                time_limit = get_sec_remaining(tic, time_limit),
            )
            length(component_obj_vals) == local_auxiliary_dimension || throw(
                DimensionMismatch(
                    "SeparableOracle: component Oracle representing subproblems " *
                    "$subproblem_indices returned $(length(component_obj_vals)) " *
                    "objective values; expected $local_auxiliary_dimension.",
                ),
            )

            local_is_in_L[k] = component_is_in_L
            sub_obj_vals[k] = component_obj_vals
            if uses_global_auxiliary_space
                hyperplanes[k] = component_hyperplanes
            else
                hyperplanes[k] = [
                    embed_local_cut(
                        h,
                        subproblem_indices,
                        auxiliary_indices,
                        dim_global_auxiliary,
                        length(x_value),
                    ) for h in component_hyperplanes
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

    cuts = vcat(hyperplanes...)
    values = vcat(sub_obj_vals...)
    is_in_L = all(local_is_in_L)

    if is_in_L
        cuts = [Hyperplane(length(x_value), dim_global_auxiliary)]
    end

    return is_in_L, cuts, values
end
