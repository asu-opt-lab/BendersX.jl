"""
    SplitOracleParam <: AbstractOracleParam

Parameters controlling [`SplitOracle`](@ref).

`SplitOracleParam` contains `dcglp_param`, which configures the DCGLP solution process, and settings for the split procedure.

# Fields

- `dcglp_param::DcglpParam`: Parameters controlling the DCGLP solution process. See [`DcglpParam`](@ref) for available options.
- `split_index_selection_rule::SplitIndexSelectionRule`: Rule used to select the master variable defining the split. See [`SplitIndexSelectionRule`](@ref) for available options.
- `disjunctive_cut_append_rule::DisjunctiveCutsAppendRule`: Rule controlling which previously generated disjunctive cuts are included in the DCGLP. See [`DisjunctiveCutsAppendRule`](@ref) for available options.
- `add_benders_cuts_to_master::Int`: Controls how byproduct Benders cuts are added to the master: unconditionally (`1`), only when violated (`2`), or not at all (`0`).
- `fraction_of_benders_cuts_to_master::Float64`: Fraction of generated Benders cuts selected for addition to the master. Must lie in `(0, 1]`.
- `reuse_dcglp::Bool`: Whether the DCGLP model is reused across oracle evaluations.
- `strengthened::Bool`: Whether strengthened disjunctive cuts are generated.
- `lift::Bool`: Whether to apply the lifting procedure for master variables whose candidate values are approximately zero or one.
- `fallback_to_typical_cuts::Bool`: Whether to fall back to typical Benders cuts when disjunctive cut generation encounters an error.
- `zero_tol::Float64`: Tolerance used when determining whether a value is effectively zero.

# Constructor

    SplitOracleParam(;
        dcglp_param = DcglpParam(),
        split_index_selection_rule = RandomFractional(),
        disjunctive_cut_append_rule = AllDisjunctiveCuts(),
        add_benders_cuts_to_master = 1,
        fraction_of_benders_cuts_to_master = 1.0,
        reuse_dcglp = true,
        strengthened = true,
        lift = false,
        fallback_to_typical_cuts = true,
        zero_tol = 1e-9,
    )

Construct split-oracle parameters with configurable DCGLP, split-selection, cut-append, and cut-generation settings.
"""
mutable struct SplitOracleParam <: AbstractOracleParam
    dcglp_param::DcglpParam
    split_index_selection_rule::SplitIndexSelectionRule
    disjunctive_cut_append_rule::DisjunctiveCutsAppendRule
    add_benders_cuts_to_master::Int
    fraction_of_benders_cuts_to_master::Float64
    reuse_dcglp::Bool
    strengthened::Bool
    lift::Bool
    fallback_to_typical_cuts::Bool
    zero_tol::Float64

    function SplitOracleParam(;
        dcglp_param::DcglpParam = DcglpParam(),
        split_index_selection_rule::SplitIndexSelectionRule = RandomFractional(),
        disjunctive_cut_append_rule::DisjunctiveCutsAppendRule = AllDisjunctiveCuts(),
        add_benders_cuts_to_master::Union{Bool,Int} = 1,
        fraction_of_benders_cuts_to_master::Float64 = 1.0,
        reuse_dcglp::Bool = true,
        strengthened::Bool = true,
        lift::Bool = false,
        fallback_to_typical_cuts::Bool = true,
        zero_tol::Float64 = 1e-9
    )
        add_benders_cuts_to_master = add_benders_cuts_to_master isa Bool ? Int(add_benders_cuts_to_master) : add_benders_cuts_to_master

        add_benders_cuts_to_master in 0:2 ||
            throw(
                ArgumentError(
                    "SplitOracleParam: `add_benders_cuts_to_master` must be true, false, " *
                    "or an integer in 0:2.",
                )
            )

        0.0 < fraction_of_benders_cuts_to_master <= 1.0 ||
            throw(
                ArgumentError(
                    "SplitOracleParam: `fraction_of_benders_cuts_to_master` must lie in (0, 1].",
                )
            )

        new(dcglp_param,
            split_index_selection_rule,
            disjunctive_cut_append_rule,
            add_benders_cuts_to_master,
            fraction_of_benders_cuts_to_master,
            reuse_dcglp,
            strengthened,
            lift,
            fallback_to_typical_cuts,
            zero_tol
        )
    end
end

"""
    SplitOracle <: AbstractDisjunctiveOracle

Split-based disjunctive Benders oracle.

`SplitOracle` uses a split disjunction and a DCGLP to generate disjunctive Benders cuts. The normalization used in the DCGLP is specified directly when constructing the oracle, while the remaining options are configured through [`SplitOracleParam`](@ref).

# Fields

- `param::SplitOracleParam`: Configuration of the split oracle.
- `normalization::AbstractNormalization`: Normalization scheme used for disjunctive cut generation. See [`AbstractNormalization`](@ref) for available options.
- `dcglp::Model`: The relaxed DCGLP problem used to generate disjunctive cuts.
- `typical_oracles::Tuple{<:AbstractOracle,<:AbstractOracle}`: Typical oracles associated with the two sides of the split.
- `dim_auxiliary::Int`: Number of auxiliary variables represented by the
  component oracles.
- `active_t_indices::Vector{Int}`: Positions in the master's auxiliary vector
  represented by the component oracles.
- `disjunctive_cuts_by_index::Vector{Vector{Hyperplane}}`: Previously generated disjunctive cuts grouped by split index.
- `disjunctive_cuts::Vector{Hyperplane}`: Collection of generated disjunctive cuts.
- `splits::Vector{Tuple{SparseVector{Float64,Int},Float64}}`: Split disjunctions generated during cut separation.

# Constructor

    SplitOracle(
        master::AbstractMaster,
        typical_oracles::Tuple{T1,T2};
        normalization::AbstractNormalization = LpDistanceNormalization(),
        param::SplitOracleParam = SplitOracleParam(),
    ) where {
    T1<:AbstractOracle,
    T2<:AbstractOracle,
}

Construct a split oracle using two typical oracles, a normalization scheme,
and the specified split-oracle configuration. The two component oracles must
have the same auxiliary-variable dimension. When both components are
`SeparableOracle`s, they must describe the same subproblems and global
auxiliary positions. The DCGLP always uses the master's complete auxiliary space;
coordinates outside those positions are not updated by the component oracles,
but remain part of the global DCGLP and returned cuts. Other component oracles
must operate on the complete master auxiliary vector.

See also: [`SplitOracleParam`](@ref), [`AbstractNormalization`](@ref)
"""
mutable struct SplitOracle{
    T1 <: AbstractOracle,
    T2 <: AbstractOracle,
    N <: AbstractNormalization,
} <: AbstractDisjunctiveOracle
    param::SplitOracleParam
    normalization::N
    dcglp::Model
    typical_oracles::Tuple{T1,T2}
    disjunctive_cuts_by_index::Vector{Vector{Hyperplane}}
    disjunctive_cuts::Vector{Hyperplane}
    splits::Vector{Tuple{SparseVector{Float64, Int}, Float64}}
    dim_auxiliary::Int
    active_t_indices::Vector{Int}

    function SplitOracle(
        master::AbstractMaster,
        typical_oracles::Tuple{T1,T2};
        normalization::AbstractNormalization = LpDistanceNormalization(),
        param::SplitOracleParam = SplitOracleParam(),
    ) where {
        T1 <: AbstractOracle,
        T2 <: AbstractOracle,
    }
        all(is_typical_oracle, typical_oracles) || throw(
            ArgumentError(
                "SplitOracle: both supplied oracles must be typical oracles."
            ),
        )

        auxiliary_dimensions = auxiliary_dimension.(typical_oracles)
        length(unique(auxiliary_dimensions)) == 1 || throw(
            DimensionMismatch(
                "SplitOracle: typical oracles must have the same " *
                "auxiliary dimension; got $(auxiliary_dimensions).",
            ),
        )
        dim_auxiliary = first(auxiliary_dimensions)

        # If either typical oracle is a SeparableOracle, both must be SeparableOracles 
        separable_typical_oracles = [oracle isa SeparableOracle for oracle in typical_oracles]

        any(separable_typical_oracles) && !all(separable_typical_oracles) && throw(
            ArgumentError(
                "SplitOracle: the two typical oracles must either both be " *
                "SeparableOracles or neither be a SeparableOracle.",
            ),
        )

        if all(separable_typical_oracles)
            first_oracle, second_oracle = typical_oracles

            first_oracle.subproblem_indices ==
            second_oracle.subproblem_indices || throw(
                ArgumentError(
                    "SplitOracle: the two typical SeparableOracles must represent the same " *
                    "subproblems; got $(first_oracle.subproblem_indices) and " *
                    "$(second_oracle.subproblem_indices)."
                ),
            )

            first_oracle.auxiliary_indices ==
            second_oracle.auxiliary_indices || throw(
                DimensionMismatch(
                    "SplitOracle: the two typical SeparableOracles must have the same " *
                    "auxiliary_indices; got $(first_oracle.auxiliary_indices) and " *
                    "$(second_oracle.auxiliary_indices)."
                ),
            )

            first_oracle.dim_global_auxiliary ==
            second_oracle.dim_global_auxiliary || throw(
                DimensionMismatch(
                    "SplitOracle: the two typical SeparableOracles must have the same " *
                    "global auxiliary dimension; got " *
                    "$(first_oracle.dim_global_auxiliary) and " *
                    "$(second_oracle.dim_global_auxiliary).",
                ),
            )

            first_oracle.dim_global_auxiliary == master.dim_t || throw(
                DimensionMismatch(
                    "SplitOracle: the typical SeparableOracle's global auxiliary dimension " *
                    "$(first_oracle.dim_global_auxiliary) must equal " *
                    "master.dim_t ($(master.dim_t)).",
                ),
            )

            active_t_indices = vcat(first_oracle.auxiliary_indices...)
        else
            dim_auxiliary == master.dim_t || throw(
                DimensionMismatch(
                    "SplitOracle: each typical oracle must represent the full auxiliary " *
                    "space when the typical oracles are not SeparableOracles; expected " *
                    "auxiliary dimension $(master.dim_t), got $dim_auxiliary.",
                ),
            )

            active_t_indices = collect(1:master.dim_t)
        end
        dcglp = build_dcglp(
            master,
            normalization,
            param;
            dim_t = master.dim_t,
        )

        disjunctive_cuts_by_index = [
            Hyperplane[] for _ in 1:master.dim_x
        ]
        disjunctive_cuts = Hyperplane[]
        splits = Tuple{SparseVector{Float64,Int},Float64}[]

        new{T1,T2,typeof(normalization)}(
            param,
            normalization,
            dcglp,
            typical_oracles,
            disjunctive_cuts_by_index,
            disjunctive_cuts,
            splits,
            dim_auxiliary,
            active_t_indices,
        )
    end
end

auxiliary_dimension(oracle::SplitOracle) = oracle.dim_auxiliary

"""
    generate_cuts(
        oracle::SplitOracle,
        x_value::Vector{Float64},
        t_value::Vector{Float64};
        tol_normalize = 1.0,
        time_limit = 3600.0,
    )

Generate disjunctive Benders cuts for a candidate master solution.

The method selects a split, updates the DCGLP for the candidate solution, and solves the resulting DCGLP.

If the configured normalization requires fallback separation, or if disjunctive cut generation encounters an error and `oracle.param.fallback_to_typical_cuts` is `true`, the first typical oracle is used to generate typical Benders cuts.

`tol_normalize` is accepted for interface compatibility with typical Benders oracles but is not used by `SplitOracle`.

# Arguments

- `oracle::SplitOracle`: Split-based disjunctive oracle.
- `x_value`: Candidate values of the master variables `x`.
- `t_value`: Candidate values of the auxiliary variables `t`.
- `tol_normalize`: Normalization factor for compatibility with the typical-oracle interface.
- `time_limit`: Maximum time allowed for cut generation, in seconds.
"""
function generate_cuts(
    oracle::SplitOracle,
    x_value::Vector{Float64},
    t_value::Vector{Float64};
    tol_normalize = 1.0, # included for interface compatibility
    time_limit::Float64 = 3600.0
)
    tic = time()

    !is_applicable(oracle.normalization, oracle, x_value, t_value) &&
        return generate_cuts(oracle.typical_oracles[1], x_value, t_value; time_limit = max(time_limit - (time() - tic), 0.0))

    update_dcglp_for_candidate!(oracle.normalization, oracle.dcglp, x_value, t_value)

    zero_indices, one_indices = choose_split_and_update_lifting!(oracle, x_value)
    update_dynamic_dcglp_constraints!(oracle)

    return solve_dcglp!(
        oracle,
        x_value,
        t_value,
        zero_indices,
        one_indices;
        time_limit = max(time_limit - (time() - tic), 0.0),
    )
end