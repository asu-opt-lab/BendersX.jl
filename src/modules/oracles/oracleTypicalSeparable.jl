"""
    (::Type{T})(
        data,
        master::AbstractMaster;
        model = update_sub_model!,
        scen_idx::Int,
        param::AbstractOracleParam,
        optimizer = DEFAULT_OPTIMIZER,
    ) where T <: AbstractTypicalOracle

Scenario-specific constructor interface used by the homogeneous convenience constructor of [`SeparableOracle`](@ref).

When calling

    SeparableOracle(data, master, T, N; ...)

`SeparableOracle` constructs one oracle of type `T` for each of the `N` subproblems. To support this form of construction, `T` must implement the constructor above. For subproblem `j`, `SeparableOracle` calls the constructor with `scen_idx = j`.

This constructor is required only for automatic homogeneous construction. It is not part of the general [`AbstractTypicalOracle`](@ref) interface. Oracles that do not implement this constructor can still be used with `SeparableOracle` by constructing them explicitly and passing them to

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
            optimizer = DEFAULT_OPTIMIZER) where T <: AbstractTypicalOracle =
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
    SeparableOracle <: AbstractTypicalOracle

Composite oracle for problems with multiple independent subproblems.

`SeparableOracle` contains one [`AbstractTypicalOracle`](@ref) for each independent subproblem and evaluates these oracles in parallel. Each sub-oracle evaluates the common linking-variable candidate and the corresponding component of the auxiliary variable `t`. The resulting cuts are embedded in the full `t` space.

Sub-oracles may have different concrete types and parameter objects.

# Fields
param::SeparableOracleParam: Parameters controlling separable evaluation.
oracles::Vector{AbstractTypicalOracle}: One configured oracle per subproblem.
N::Int: Number of independent subproblems.

# Constructors

    SeparableOracle(
        master::Master,
        oracles::AbstractVector{<:AbstractTypicalOracle};
        param = SeparableOracleParam(),
    )

Construct a `SeparableOracle` from already configured sub-oracles.

    SeparableOracle(
        data,
        master::Master,
        oracle_type::Type{T},
        N::Int;
        model = update_sub_model!,
        sub_oracle_param = BasicOracleParam(),
        param = SeparableOracleParam(),
        optimizer = DEFAULT_OPTIMIZER,
    ) where T <: AbstractTypicalOracle

Convenience constructor for the common case in which all `N` subproblems use the same oracle type and configuration. It constructs one oracle for each subproblem using `scen_idx = 1:N`. The same `sub_oracle_param` and `model` function are passed to every sub-oracle.

`N` must equal `master.dim_t`, with one component of the master auxiliary variable `t` associated with each subproblem.

# Throws

Throws a `DimensionMismatch` if `N != master.dim_t`.

See also: [`AbstractOracle`](@ref), [`generate_cuts`](@ref)
"""
mutable struct SeparableOracle <: AbstractTypicalOracle
    param::SeparableOracleParam 

    oracles::Vector{AbstractTypicalOracle}
    N::Int

    function SeparableOracle(
        master::Master,
        oracles::AbstractVector{<:AbstractTypicalOracle};
        param::SeparableOracleParam = SeparableOracleParam(),
    )
        N = length(oracles)

        N == master.dim_t || throw(
            DimensionMismatch(
                "SeparableOracle: number of sub-oracles ($N) must equal " *
                "master.dim_t ($(master.dim_t))."
            )
        )

        @info "SeparableOracle: N=$N subproblems, " *
              "$(Threads.nthreads()) threads available for parallel execution"

        new(
            param,
            AbstractTypicalOracle[oracles...],
            N,
        )
    end
end

function SeparableOracle(
        data,
        master::Master,
        oracle_type::Type{T},
        N::Int;
        model = update_sub_model!,
        sub_oracle_param::AbstractOracleParam = BasicOracleParam(),
        param::SeparableOracleParam = SeparableOracleParam(),
        optimizer = DEFAULT_OPTIMIZER,
        ) where {T <: AbstractTypicalOracle}

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

"""
    generate_cuts(
        oracle::SeparableOracle,
        x_value::Vector{Float64},
        t_value::Vector{Float64};
        tol_normalize = 1.0,
        time_limit = 3600.0,
    )

Generate Benders cuts by evaluating all sub-oracles in parallel.

Each sub-oracle is evaluated at the common candidate `x_value` and its corresponding component `t_value[j]`. Generated cuts are expanded to the full `t` dimension and associated with the corresponding subproblem.

If any sub-oracle separates the candidate, the generated cuts and subproblem objective values are returned collectively. Otherwise, the candidate is reported as belonging to the separable oracle's feasible region.
"""
function generate_cuts(oracle::SeparableOracle, x_value::Vector{Float64}, t_value::Vector{Float64}; tol_normalize = 1.0, time_limit = 3600.0)
    tic = time()
    N = oracle.N
    is_in_L = Vector{Bool}(undef,N)
    sub_obj_val = Vector{Vector{Float64}}(undef,N)
    hyperplanes = Vector{Vector{Hyperplane}}(undef,N)

    try
        Threads.@threads for j=1:N
            is_in_L[j], hyperplanes[j], sub_obj_val[j] = generate_cuts(oracle.oracles[j], x_value, [t_value[j]], tol_normalize = tol_normalize; time_limit = get_sec_remaining(tic, time_limit))

            # correct dimension for t_j's
            for h in hyperplanes[j]
                coeff_t = h.a_t[1]
                h.a_t = spzeros(length(t_value))
                h.a_t[j] = coeff_t
            end
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
        return true, [Hyperplane(length(x_value), length(t_value))], deepcopy(t_value)
    end
end
