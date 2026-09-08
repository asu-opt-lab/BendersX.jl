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
            default = "ga250a-1"
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
            default = 14400.0
        "--preprocessing_time_limit"
            help = "Preprocessing time limit in seconds"
            arg_type = Float64
            default = 100.0
        "--threads"
            help = "Number of CPLEX threads"
            arg_type = Int
            default = 7
        "--build_only"
            help = "Build the Benders environment without solving"
            arg_type = Bool
            default = false
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
threads = args["threads"]
build_only = args["build_only"]

@info "UFLP Benders BnB knapsack script" instance = instance seed = args["seed"] time_limit = time_limit preprocessing_time_limit = preprocessing_time_limit threads = threads build_only = build_only

# -----------------------------------------------------------------------------
# load problem data
# -----------------------------------------------------------------------------
data = read_Simple_data(instance)

# -----------------------------------------------------------------------------
# Customize master model function for knapsack oracle (needs t[1:n_customers])
# -----------------------------------------------------------------------------
function update_master_model!(model::Model, data::UFLPData)
    optimizer = optimizer_with_attributes(CPLEX.Optimizer,
        "CPXPARAM_Threads" => threads, "CPX_PARAM_EPINT" => 1e-9,
        "CPX_PARAM_EPRHS" => 1e-9, "CPX_PARAM_EPGAP" => 1e-6,
        MOI.Silent() => true)
    set_optimizer(model, optimizer)
    @variable(model, x[1:data.n_facilities], Bin)
    @variable(model, t[1:data.n_customers] >= -1e6)
    @constraint(model, sum(x) >= 2)
    @objective(model, Min, data.fixed_costs'* x + sum(t))
    return (x = x, ), t
end

# -----------------------------------------------------------------------------
# load parameters
# -----------------------------------------------------------------------------
# Algorithm parameters
benders_param = BendersBnBParam(
    time_limit = time_limit,
    gap_tolerance = 1e-6,
    verbose = true
)

# -----------------------------------------------------------------------------
# master model
# -----------------------------------------------------------------------------
master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
set_optimizer_attribute(master.model, "CPX_PARAM_BRDIR", 1)

# -----------------------------------------------------------------------------
# typical oracle
# -----------------------------------------------------------------------------
typical_oracle = UFLKnapsackOracle(
    data;
    param = UFLKnapsackOracleParam(add_only_violated_cuts = true),
)

# -----------------------------------------------------------------------------
# Benders preprocessing
# -----------------------------------------------------------------------------
preprocessing_seq_type = BendersSeq
preprocessing_seq_param = BendersSeqParam(
    time_limit = min(preprocessing_time_limit, time_limit),
    gap_tolerance = 1e-9,
    verbose = true
)

# Create Benders preprocessing with oracle
preprocessing = LPRelaxationPreprocessing(typical_oracle; seq_env_type = preprocessing_seq_type, param = preprocessing_seq_param)

# -----------------------------------------------------------------------------
# lazy callback
# -----------------------------------------------------------------------------
lazy_callback = LazyCallback(typical_oracle)

# -----------------------------------------------------------------------------
# user callback
# -----------------------------------------------------------------------------
user_callback = NoUserCallback()

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
if build_only
    @info "UFLP Benders BnB knapsack script build completed without solve." instance = instance
else
    solution_log = solve!(env)
    obj_value = try
        env.obj_value
    catch
        NaN
    end
    @info "UFLP Benders BnB knapsack script finished" instance = instance termination_status = env.termination_status objective_value = obj_value
end
