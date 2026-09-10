module BendersX

using JuMP
using MathOptInterface
using Printf
using LinearAlgebra
using SparseArrays
using DataFrames
using GLPK

const MOI = MathOptInterface

include("types.jl")
include("utils/utils.jl")
include("modules/modules.jl")
include("artifact_utils.jl")
include("problems/problems.jl")

# Public API surface is centralized here. Child files may define symbols, but
# they must not decide visibility via `export` or `public`.

# Default `using BendersX` workflow
export Master
export BendersSeq, BendersSeqInOut, BendersBnB
export ClassicalOracle, UnifiedOracle, ParetoOracle, SeparableOracle
export SplitOracle
export solve!

# User-facing parameter and configuration types
export BasicOracleParam, ClassicalOracleParam, UnifiedOracleParam, ParetoOracleParam
export SeparableOracleParam, DcglpParam
export SplitOracleParam
export LpDistanceNormalization
export ReversePolarNormalization
export BendersSeqParam, BendersSeqInOutParam, BendersBnBParam
export LazyCallback, UserCallback, NoUserCallback, UserCallbackParam
export LPRelaxationPreprocessing, NoPreprocessing, DisjunctiveLPRelaxationPreprocessing

# Common runtime statuses and configuration values
export TerminationStatus, NotSolved, TimeLimit, Optimal, InfeasibleOrNumericalIssue
export GBCBoundType, UpperBound, LowerBound, Fixed
export RandomFractional, MostFractional, LargestFractional
export NoDisjunctiveCuts, AllDisjunctiveCuts, DisjunctiveCutsSmallerIndices

# Public extension interfaces
public AbstractBendersEnv, AbstractBendersSeq, AbstractBendersBnB
public AbstractMaster
public AbstractOracle, AbstractOracleParam, AbstractTypicalOracle, AbstractDisjunctiveOracle
public AbstractPreprocessing
public AbstractNormalization
public AbstractLazyCallback, AbstractUserCallback
public AbstractLoopState, AbstractLoopLog, AbstractLoopParam
public AbstractBendersSeqState, AbstractBendersSeqLog, AbstractBendersSeqParam
public AbstractBendersBnBState, AbstractBendersBnBLog, AbstractBendersBnBParam
public SplitIndexSelectionRule, DisjunctiveCutsAppendRule
public generate_cuts, add_normalization_constraint!, update_dcglp_upper_bound_and_gap!
public disjunctive_cut_normalization_value
public callback_node_count, callback_node_depth
export update_master_model!, update_sub_model!

# Public but not auto-imported advanced utilities and support types
public Hyperplane, aggregate, evaluate_violation, select_top_fraction
public hyperplanes_to_expression, add_constraints
public copy_variables!, var_from_tuple, transfer_scaled_linear_rows_and_bounds_with_types!
public infeasibility_report
public TimeLimitException, UnexpectedModelStatusException, UnimplementedInterfaceException
public AlgorithmException, UnsupportedModelException

# Problem-specific helpers available via `using BendersX`
export CFLPData, UFLPData, SCFLPData, SNIPData
export CFLKnapsackOracle, CFLKnapsackOracleParam
export UFLKnapsackOracle, UFLKnapsackOracleParam
export read_GK_data, read_cfl_file, read_cflp_benchmark_data
export read_uflp_benchmark_data, read_Simple_data
export read_stochastic_capacited_facility_location_problem, read_snip_data
export update_knapsack_master_model!, update_sub_gbc_model!

end # module BendersX
