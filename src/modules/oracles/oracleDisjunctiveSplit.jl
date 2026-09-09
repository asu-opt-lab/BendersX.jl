"""
    SplitOracleParam <: AbstractOracleParam

Parameters controlling [`SplitOracle`](@ref).

`SplitOracleParam` contains settings for the split/disjunctive procedure. The
normalization algorithm is configured directly on [`SplitOracle`](@ref).

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

Construct split-oracle parameters with configurable DCGLP, split-selection,
cut-append, and cut-generation settings.
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

`SplitOracle` uses a split disjunction and a DCGLP to generate disjunctive
Benders cuts. The normalization scheme is a direct subcomponent of the oracle,
while the remaining procedure settings are specified through
[`SplitOracleParam`](@ref).

# Fields

- `param::SplitOracleParam`: Configuration of the split oracle.
- `normalization::AbstractNormalization`: Normalization scheme used for disjunctive cut generation. See [`AbstractNormalization`](@ref) for available options.
- `dcglp::Model`: The relaxed DCGLP problem used to generate disjunctive cuts.
- `typical_oracles::Tuple{<:AbstractTypicalOracle,<:AbstractTypicalOracle}`: Typical Benders oracles associated with the two sides of the split.
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
    T1<:AbstractTypicalOracle,
    T2<:AbstractTypicalOracle,
}

Construct a split oracle using two typical Benders oracles, a normalization
algorithm, and the specified split-oracle configuration.

See also: [`SplitOracleParam`](@ref), [`AbstractNormalization`](@ref)
"""
mutable struct SplitOracle{
    T1 <: AbstractTypicalOracle,
    T2 <: AbstractTypicalOracle,
    N <: AbstractNormalization,
} <: AbstractDisjunctiveOracle
    param::SplitOracleParam
    normalization::N
    dcglp::Model
    typical_oracles::Tuple{T1,T2}
    disjunctive_cuts_by_index::Vector{Vector{Hyperplane}}
    disjunctive_cuts::Vector{Hyperplane}
    splits::Vector{Tuple{SparseVector{Float64, Int}, Float64}}

    function SplitOracle(
        master::AbstractMaster,
        typical_oracles::Tuple{T1,T2};
        normalization::AbstractNormalization = LpDistanceNormalization(),
        param::SplitOracleParam = SplitOracleParam(),
    ) where {
        T1<:AbstractTypicalOracle,
        T2<:AbstractTypicalOracle,
    }
        dcglp = build_dcglp(master, normalization, param)
        
        disjunctive_cuts_by_index = [
            Hyperplane[] for _ in 1:master.dim_x
        ]
        disjunctive_cuts = Hyperplane[]
        splits = Tuple{SparseVector{Float64,Int},Float64}[]

        new{T1, T2, typeof(normalization)}(
            param,
            normalization,
            dcglp,
            typical_oracles,
            disjunctive_cuts_by_index,
            disjunctive_cuts,
            splits
        )
    end
end

"""
    generate_cuts(
        oracle::SplitOracle,
        x_value::Vector{Float64},
        t_value::Vector{Float64};
        tol_normalize = 1.0,
        time_limit = 3600.0,
    )

Generate disjunctive Benders cuts for a candidate master solution.

The method selects a split, updates the DCGLP for the candidate solution, and
solves the resulting DCGLP using `oracle.normalization`.

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
    tol_normalize = 1.0, # just to match the signature of the typical oracle
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
