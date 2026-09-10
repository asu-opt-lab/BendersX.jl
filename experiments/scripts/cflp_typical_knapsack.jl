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
            help = "Total time limit in seconds"
            arg_type = Float64
            default = 14400.0
        "--preprocessing_time_limit"
            help = "Preprocessing time limit in seconds"
            arg_type = Float64
            default = 300.0
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

# -----------------------------------------------------------------------------
# load problem data
# -----------------------------------------------------------------------------
data = read_cfl_file(instance)

# -----------------------------------------------------------------------------
# master model
# -----------------------------------------------------------------------------
master = Master(data; model = update_master_model!, optimizer = mip_optimizer)

# -----------------------------------------------------------------------------
# typical oracle
# -----------------------------------------------------------------------------
typical_oracle = CFLKnapsackOracle(data, master; model = update_sub_model!, optimizer = optimizer)

# -----------------------------------------------------------------------------
# BendersSeqInOut
# -----------------------------------------------------------------------------
start_time = time()
undo = relax_integrality(master.model)
benders_inout_param = BendersSeqInOutParam(;
            time_limit = preprocessing_time_limit,
            gap_tolerance = 1e-6,
            verbose = true,
            stabilizing_x = ones(data.n_facilities),
            α = 0.9,
            λ = 0.1
            )
set_optimizer_attribute(typical_oracle.model, "CPX_PARAM_LPMETHOD", 2)
env = BendersSeqInOut(master, typical_oracle; param = benders_inout_param)
solution_log = solve!(env)
set_optimizer_attribute(typical_oracle.model, "CPX_PARAM_LPMETHOD", 0)
println("BendersSeqInOut done")
undo()
spend_time_prev = time() - start_time
println("Spend time: $spend_time_prev seconds")

# -----------------------------------------------------------------------------
# BendersBnB
# -----------------------------------------------------------------------------
preprocessing = NoPreprocessing()
lazy_callback = LazyCallback(typical_oracle)
user_callback = NoUserCallback()

benders_param = BendersBnBParam(
    time_limit = max(time_limit - spend_time_prev, 0.0),
    gap_tolerance = 1e-6,
    verbose = true
)
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
println("BendersBnB done")
@info "total time: $(time() - start_time) seconds"
