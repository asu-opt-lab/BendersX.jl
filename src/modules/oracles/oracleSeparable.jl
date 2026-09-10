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

`SeparableOracle` constructs one oracle of type `T` for each of the `N` subproblems. To support this form of construction, `T` must implement the constructor above. For subproblem `j`, `SeparableOracle` calls the constructor with `scen_idx = j`.

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

This parameter container is currently empty and serves as an extension point for future controls related to scenario handling or parallel evaluation.

Parameters of the individual sub-oracles are stored by those oracles rather than by `SeparableOracle`.

See also: [`SeparableOracle`](@ref), [`AbstractOracleParam`](@ref)
"""
mutable struct SeparableOracleParam <: AbstractOracleParam
    # may contain parameters for scenario handling.
end

"""
    SeparableOracle <: AbstractOracle

Composite oracle for problems with multiple independent subproblems.

`SeparableOracle` contains one [`AbstractOracle`](@ref) for each independent
subproblem and evaluates these oracles in parallel. Each sub-oracle evaluates
the common linking-variable candidate and its corresponding block of auxiliary
variables. The resulting cuts are copied and embedded in the full auxiliary
space without modifying the child oracle's cuts.

Sub-oracles may have different concrete types and parameter objects.

# Fields
param::SeparableOracleParam: Parameters controlling separable evaluation.
oracles::Vector{AbstractOracle}: One configured oracle per subproblem.
auxiliary_ranges::Vector{UnitRange{Int}}: Position of each sub-oracle's
auxiliary-variable block in the full auxiliary space.
dim_auxiliary::Int: Total auxiliary-variable dimension represented by all
sub-oracles.

# Constructors

    SeparableOracle(
        master::AbstractMaster,
        oracles::AbstractVector{<:AbstractOracle};
        param = SeparableOracleParam(),
    )

Construct a `SeparableOracle` from already configured sub-oracles.

    SeparableOracle(
        data,
        master::AbstractMaster,
        oracle_type::Type{T},
        N::Int;
        model = update_sub_model!,
        sub_oracle_param = BasicOracleParam(),
        param = SeparableOracleParam(),
        optimizer = DEFAULT_OPTIMIZER,
    ) where T <: AbstractOracle

Convenience constructor for the common case in which all `N` subproblems use
the same oracle type and configuration. It constructs one oracle for each
subproblem using `scen_idx = 1:N`. The same `sub_oracle_param` and `model`
function are passed to every sub-oracle.

The sum of the sub-oracles' [`auxiliary_dimension`](@ref) values must equal
`master.dim_t`. Scalar sub-oracles therefore use one auxiliary variable each,
while a sub-oracle such as `UFLKnapsackOracle` may own a multi-variable block.

# Throws

Throws a `DimensionMismatch` if the total auxiliary-variable dimension of the
sub-oracles does not equal `master.dim_t`.

See also: [`AbstractOracle`](@ref), [`generate_cuts`](@ref)
"""
mutable struct SeparableOracle <: AbstractOracle
    param::SeparableOracleParam 

    oracles::Vector{AbstractOracle}
    auxiliary_ranges::Vector{UnitRange{Int}}
    dim_auxiliary::Int

    function SeparableOracle(
        master::AbstractMaster,
        oracles::AbstractVector{<:AbstractOracle};
        param::SeparableOracleParam = SeparableOracleParam(),
    )
        isempty(oracles) && throw(
            ArgumentError("SeparableOracle: at least one sub-oracle is required."),
        )

        child_dimensions = auxiliary_dimension.(oracles)
        all(>(0), child_dimensions) || throw(
            ArgumentError(
                "SeparableOracle: sub-oracle auxiliary dimensions must be " *
                "positive; got $(child_dimensions).",
            ),
        )

        dim_auxiliary = sum(child_dimensions)
        dim_auxiliary == master.dim_t || throw(
            DimensionMismatch(
                "SeparableOracle: sub-oracles represent $dim_auxiliary " *
                "auxiliary variables, but " *
                "master.dim_t ($(master.dim_t))."
            )
        )

        auxiliary_ends = cumsum(child_dimensions)
        auxiliary_ranges = UnitRange{Int}[
            (auxiliary_ends[j] - child_dimensions[j] + 1):auxiliary_ends[j]
            for j in eachindex(child_dimensions)
        ]

        @info "SeparableOracle: N=$(length(oracles)) subproblems, " *
              "dim_auxiliary=$dim_auxiliary, " *
              "$(Threads.nthreads()) threads available for parallel execution"

        new(
            param,
            AbstractOracle[oracles...],
            auxiliary_ranges,
            dim_auxiliary,
        )
    end
end

function SeparableOracle(
        data,
        master::AbstractMaster,
        oracle_type::Type{T},
        N::Int;
        model = update_sub_model!,
        sub_oracle_param::AbstractOracleParam = BasicOracleParam(),
        param::SeparableOracleParam = SeparableOracleParam(),
        optimizer = DEFAULT_OPTIMIZER,
        ) where {T <: AbstractOracle}

            oracles = [
                oracle_type(
                    data,
                    master;
                    model = model,
                    scen_idx = j,
                    param = deepcopy(sub_oracle_param),
                    optimizer = optimizer,
                )
                for j in 1:N
            ]

            return SeparableOracle(
                master,
                oracles;
                param = param,
            )
end

is_typical_oracle(oracle::SeparableOracle) =
    all(is_typical_oracle, oracle.oracles)

auxiliary_dimension(oracle::SeparableOracle) = oracle.dim_auxiliary

function embed_local_cut(
    h::Hyperplane,
    child_index::Int,
    auxiliary_range::UnitRange{Int},
    dim_auxiliary::Int,
    dim_x::Int,
)
    length(h.a_x) == dim_x || throw(
        DimensionMismatch(
            "SeparableOracle child $child_index returned a cut with " *
            "a_x length $(length(h.a_x)); expected $dim_x.",
        ),
    )
    child_dimension = length(auxiliary_range)
    length(h.a_t) == child_dimension || throw(
        DimensionMismatch(
            "SeparableOracle child $child_index returned a cut with " *
            "a_t length $(length(h.a_t)); expected $child_dimension.",
        ),
    )

    embedded = Hyperplane(dim_x, dim_auxiliary)
    embedded.a_x = copy(h.a_x)
    local_indices, local_values = findnz(h.a_t)
    offset = first(auxiliary_range) - 1
    for k in eachindex(local_indices)
        embedded.a_t[offset + local_indices[k]] = local_values[k]
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

Generate Benders cuts by evaluating all sub-oracles in parallel.

Each sub-oracle is evaluated at the common candidate `x_value` and its
corresponding block of `t_value`. Generated cuts are expanded to the full `t`
dimension and associated with the corresponding subproblem. Sub-oracles may
represent different positive auxiliary-variable dimensions.

If any sub-oracle separates the candidate, the generated cuts and subproblem objective values are returned collectively. Otherwise, the candidate is reported as belonging to the separable oracle's feasible region.
"""
function generate_cuts(oracle::SeparableOracle, x_value::Vector{Float64}, t_value::Vector{Float64}; tol_normalize = 1.0, time_limit = 3600.0)
    tic = time()
    N = length(oracle.oracles)
    dim_auxiliary = auxiliary_dimension(oracle)
    length(t_value) == dim_auxiliary || throw(
        DimensionMismatch(
            "SeparableOracle received t_value with length $(length(t_value)); " *
            "expected $dim_auxiliary.",
        ),
    )
    is_in_L = Vector{Bool}(undef,N)
    sub_obj_val = Vector{Vector{Float64}}(undef,N)
    hyperplanes = Vector{Vector{Hyperplane}}(undef,N)

    try
        Threads.@threads for j=1:N
            auxiliary_range = oracle.auxiliary_ranges[j]
            child_dimension = length(auxiliary_range)
            child_is_in_L, child_hyperplanes, child_obj_val = generate_cuts(
                oracle.oracles[j],
                x_value,
                t_value[auxiliary_range];
                tol_normalize = tol_normalize,
                time_limit = get_sec_remaining(tic, time_limit),
            )
            length(child_obj_val) == child_dimension || throw(
                DimensionMismatch(
                    "SeparableOracle child $j returned $(length(child_obj_val)) " *
                    "objective values; expected $child_dimension.",
                ),
            )

            is_in_L[j] = child_is_in_L
            sub_obj_val[j] = child_obj_val
            hyperplanes[j] = [
                embed_local_cut(
                    h,
                    j,
                    auxiliary_range,
                    dim_auxiliary,
                    length(x_value),
                ) for h in child_hyperplanes
            ]
        end
    catch err
        err isa CompositeException || rethrow()
        task_failure = first(err.exceptions)
        task_failure isa TaskFailedException || rethrow()
        failures = current_exceptions(task_failure.task; backtrace = false)
        isempty(failures) && rethrow()
        failure = first(failures)
        throw(failure.exception)
    end

    if any(.!is_in_L)
        return false, reduce(vcat, hyperplanes), reduce(vcat, sub_obj_val)
    else
        return true, [Hyperplane(length(x_value), length(t_value))], reduce(vcat, sub_obj_val)
    end
end
