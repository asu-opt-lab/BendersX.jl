using BendersX
using JuMP
using CPLEX
using ArgParse
using Printf
using Random

include(normpath(joinpath(@__DIR__, "..", "solver_defaults.jl")))

function parse_commandline()
    settings = ArgParseSettings()
    @add_arg_table! settings begin
        "--instance"
            help = "UFLP instance name"
            arg_type = String
            default = "ga250a-2"
        "--seed"
            help = "Random seed"
            arg_type = Int
            default = 1
        "--output_dir"
            help = "Output directory kept for compatibility with older launchers"
            arg_type = String
            default = "output"
        "--time_limit"
            help = "Benders branch-and-bound time limit in seconds"
            arg_type = Float64
            default = 14400.0
        "--dcglp_time_limit"
            help = "DCGLP separation time limit in seconds"
            arg_type = Float64
            default = 100.0
        "--dcglp_iter_limit"
            help = "DCGLP iteration limit"
            arg_type = Int
            default = 250
        "--dcglp_halt_limit"
            help = "DCGLP halt limit"
            arg_type = Int
            default = 3
        "--frequency"
            help = "User callback frequency"
            arg_type = Int
            default = 500
        "--threads"
            help = "Number of CPLEX threads"
            arg_type = Int
            default = 7
        "--reuse_dcglp"
            help = "Reuse the DCGLP between callback calls"
            arg_type = Bool
            default = false
        "--strengthened"
            help = "Use strengthened split cuts"
            arg_type = Bool
            default = true
        "--lift"
            help = "Lift disjunctive cuts"
            arg_type = Bool
            default = false
        "--build_only"
            help = "Build the Benders environment without solving"
            arg_type = Bool
            default = false
    end
    return parse_args(settings)
end

args = parse_commandline()

instance = args["instance"]
seed = args["seed"]
time_limit = args["time_limit"]
dcglp_time_limit = args["dcglp_time_limit"]
dcglp_iter_limit = args["dcglp_iter_limit"]
dcglp_halt_limit = args["dcglp_halt_limit"]
frequency = args["frequency"]
threads = args["threads"]
reuse_dcglp = args["reuse_dcglp"]
strengthened = args["strengthened"]
lift = args["lift"]
build_only = args["build_only"]

Random.seed!(seed)

@info "vertical_reverse_polar hard UFL script" instance = instance seed = seed time_limit = time_limit frequency = frequency threads = threads reuse_dcglp = reuse_dcglp strengthened = strengthened lift = lift build_only = build_only

data = read_Simple_data(instance)

function customize_master_knapsack!(model::Model, data::UFLPData)
    optimizer = optimizer_with_attributes(
        CPLEX.Optimizer,
        "CPXPARAM_Threads" => threads,
        "CPX_PARAM_EPINT" => 1e-9,
        "CPX_PARAM_EPRHS" => 1e-9,
        "CPX_PARAM_EPGAP" => 1e-6,
        MOI.Silent() => true,
    )
    set_optimizer(model, optimizer)
    @variable(model, x[1:data.n_facilities], Bin)
    @variable(model, t[1:data.n_customers] >= -1e6)
    @constraint(model, sum(x) >= 2)
    @objective(model, Min, data.fixed_costs' * x + sum(t))
    return (x = x,), t
end

benders_param = BendersBnBParam(
    time_limit = time_limit,
    gap_tolerance = 1e-6,
    verbose = true,
)

dcglp_optimizer = optimizer_with_attributes(
    CPLEX.Optimizer,
    "CPXPARAM_Threads" => threads,
    "CPX_PARAM_EPRHS" => 1e-9,
    "CPX_PARAM_NUMERICALEMPHASIS" => 1,
    "CPX_PARAM_EPOPT" => 1e-9,
    MOI.Silent() => true,
)
dcglp_param = DcglpParam(;
    optimizer = dcglp_optimizer,
    time_limit = dcglp_time_limit,
    gap_tolerance = 1e-3,
    halt_limit = dcglp_halt_limit,
    iter_limit = dcglp_iter_limit,
    verbose = false,
)

oracle_param = SplitOracleParam(;
    dcglp_param = dcglp_param,
    split_index_selection_rule = LargestFractional(),
    disjunctive_cut_append_rule = AllDisjunctiveCuts(),
    add_benders_cuts_to_master = 2,
    fraction_of_benders_cuts_to_master = 0.05,
    reuse_dcglp = reuse_dcglp,
    strengthened = strengthened,
    lift = lift,
    zero_tol = 1e-9,
)

master = Master(data; model = customize_master_knapsack!, optimizer = mip_optimizer)
set_optimizer_attribute(master.model, "CPX_PARAM_BRDIR", 1)

typical_oracles = (
    UFLKnapsackOracle(
        data;
        param = UFLKnapsackOracleParam(add_only_violated_cuts = true),
    ),
    UFLKnapsackOracle(
        data;
        param = UFLKnapsackOracleParam(add_only_violated_cuts = true),
    ),
)
disjunctive_oracle = SplitOracle(
    master,
    typical_oracles;
    normalization = ReversePolarNormalization(),
    param = oracle_param,
)

lazy_oracle = UFLKnapsackOracle(
    data;
    param = UFLKnapsackOracleParam(add_only_violated_cuts = true),
)
preprocessing = LPRelaxationPreprocessing(
    lazy_oracle;
    seq_env_type = BendersSeq,
    param = BendersSeqParam(
        time_limit = min(100.0, time_limit),
        gap_tolerance = 1e-9,
        verbose = true,
    ),
)
lazy_callback = LazyCallback(lazy_oracle)
user_callback = UserCallback(disjunctive_oracle; param = UserCallbackParam(frequency = frequency))

env = BendersBnB(
    master;
    preprocessing = preprocessing,
    lazy_callback = lazy_callback,
    user_callback = user_callback,
    param = benders_param,
)

if build_only
    @info "vertical_reverse_polar hard UFL script build completed without solve." instance = instance
else
    solve!(env)
    obj_value = try
        env.obj_value
    catch
        NaN
    end
    @info "vertical_reverse_polar hard UFL script finished" instance = instance termination_status = env.termination_status objective_value = obj_value
end
