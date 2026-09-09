using JuMP, DataFrames, Logging, CSV
using BendersX
using ArgParse
using Random
using Printf
using Statistics
using CPLEX
include(normpath(joinpath(@__DIR__, "..", "solver_defaults.jl")))

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--instance"
            help = "Instance name"
            arg_type = String
            default = "T100x100_5_1"
        "--seed"
            help = "Random seed"
            arg_type = Int
            default = 1
        "--output_dir"
            help = "Output directory"
            arg_type = String
            default = "output"
        "--time_limit"
            help = "Benders branch-and-bound time limit in seconds"
            arg_type = Float64
            default = 200.0
        "--preprocessing_time_limit"
            help = "Preprocessing time limit in seconds"
            arg_type = Float64
            default = 100.0
        "--frequency"
            help = "User callback frequency"
            arg_type = Int
            default = 250
    end
    return parse_args(s)
end

# load settings
args = parse_commandline()

Random.seed!(args["seed"])
instance = args["instance"]
output_dir = args["output_dir"]
time_limit = args["time_limit"]
preprocessing_time_limit = args["preprocessing_time_limit"]
frequency = args["frequency"]

# -----------------------------------------------------------------------------
# load problem data
# -----------------------------------------------------------------------------
data = read_cfl_file(instance)

# -----------------------------------------------------------------------------
# load parameters
# -----------------------------------------------------------------------------
# Algorithm parameters
benders_param = BendersBnBParam(
    time_limit = time_limit,
    gap_tolerance = 1e-6,
    verbose = true
)

dcglp_optimizer = optimizer_with_attributes(CPLEX.Optimizer,
    "CPX_PARAM_EPRHS" => 1e-9, "CPX_PARAM_NUMERICALEMPHASIS" => 1,
    "CPX_PARAM_EPOPT" => 1e-9, MOI.Silent() => true)
dcglp_param = DcglpParam(; optimizer = dcglp_optimizer,
    time_limit = 1000.0,
    gap_tolerance = 1e-3,
    halt_limit = 3,
    iter_limit = 250,
    verbose = true
)

oracle_param = SplitOracleParam(; dcglp_param = dcglp_param,
    split_index_selection_rule = RandomFractional(),
    disjunctive_cut_append_rule = AllDisjunctiveCuts(),
    strengthened = true,
    add_benders_cuts_to_master = true,
    fraction_of_benders_cuts_to_master = 0.5,
    reuse_dcglp = true
)

# -----------------------------------------------------------------------------
# master model
# -----------------------------------------------------------------------------
master = Master(data; model = update_master_model!, optimizer = mip_optimizer)

# -----------------------------------------------------------------------------
# typical oracles
# -----------------------------------------------------------------------------
# Create two oracles for kappa & nu
typical_oracles = [
    CFLKnapsackOracle(data, master; model = update_sub_model!, optimizer = optimizer),
    CFLKnapsackOracle(data, master; model = update_sub_model!, optimizer = optimizer)
]

# -----------------------------------------------------------------------------
# disjunctive oracle
# -----------------------------------------------------------------------------
disjunctive_oracle = SplitOracle(
    master,
    Tuple(typical_oracles);
    normalization = LpDistanceNormalization(1.0),
    param = oracle_param,
)

# -----------------------------------------------------------------------------
# Benders preprocessing
# -----------------------------------------------------------------------------
lazy_oracle = CFLKnapsackOracle(data, master; model = update_sub_model!, optimizer = optimizer)

preprocessing_seq_type = BendersSeqInOut
preprocessing_seq_param = BendersSeqInOutParam(
    time_limit = preprocessing_time_limit,
    gap_tolerance = 1e-6,
    stabilizing_x = ones(data.n_facilities),
    α = 0.9,
    λ = 0.1,
    verbose = true
)

# Create Benders preprocessing with oracle
preprocessing = LPRelaxationPreprocessing(lazy_oracle; seq_env_type = preprocessing_seq_type, param = preprocessing_seq_param)

# -----------------------------------------------------------------------------
# lazy callback
# -----------------------------------------------------------------------------
lazy_callback = LazyCallback(lazy_oracle)

# -----------------------------------------------------------------------------
# user callback
# -----------------------------------------------------------------------------
user_callback = UserCallback(disjunctive_oracle; param=UserCallbackParam(frequency=frequency))

# -----------------------------------------------------------------------------
# BendersBnB
# -----------------------------------------------------------------------------
env = BendersBnB(
    master;
    preprocessing = preprocessing,
    lazy_callback = lazy_callback,
    user_callback = user_callback,
    param = benders_param,
)

# -----------------------------------------------------------------------------
# solve
# -----------------------------------------------------------------------------
solution_log = solve!(env)
