using BendersX
using CSV
using CPLEX
using DataFrames
using JuMP
using Random
using Test

include(normpath(joinpath(@__DIR__, "..", "solver_defaults.jl")))

function build_local_split_oracle(
    data::SUFLPData,
    master::Master,
    scenario::Int,
    oracle_param::BendersX.AbstractOracleParam,
    split_param::SplitOracleParam,
)
    kappa = UFLKnapsackOracle(
        data;
        scen_idx = scenario,
        param = deepcopy(oracle_param),
    )
    nu = UFLKnapsackOracle(
        data;
        scen_idx = scenario,
        param = deepcopy(oracle_param),
    )

    return SplitOracle(
        master,
        (kappa, nu);
        normalization = LpDistanceNormalization(1.0),
        param = deepcopy(split_param),
    )
end

@testset verbose = true "SUFLP Sequential Separable SplitOracle Tests" begin
    reference_path = normpath(
        joinpath(@__DIR__, "..", "reference_objectives", "suflp.csv"),
    )
    reference_df = DataFrame(CSV.File(reference_path))
    @assert(
        nrow(reference_df) == length(unique(reference_df.instance_name)),
        "Duplicate SUFLP reference objectives found in $reference_path.",
    )
    reference_objectives = Dict(
        String(row.instance_name) => Float64(row.objective_value)
        for row in eachrow(reference_df)
    )

    benders_param = BendersSeqParam(;
        time_limit = 2000.0,
        gap_tolerance = 1.0e-6,
        verbose = false,
    )
    dcglp_optimizer = optimizer_with_attributes(
        CPLEX.Optimizer,
        "CPXPARAM_Threads" => 7,
        "CPX_PARAM_EPRHS" => 1.0e-9,
        "CPX_PARAM_NUMERICALEMPHASIS" => 1,
        "CPX_PARAM_EPOPT" => 1.0e-9,
        MOI.Silent() => true,
    )
    dcglp_param = DcglpParam(;
        optimizer = dcglp_optimizer,
        time_limit = 1000.0,
        gap_tolerance = 1.0e-3,
        halt_limit = 3,
        iter_limit = 250,
        verbose = false,
    )
    split_param = SplitOracleParam(;
        dcglp_param = dcglp_param,
        split_index_selection_rule = RandomFractional(),
        disjunctive_cut_append_rule = AllDisjunctiveCuts(),
        strengthened = true,
        add_benders_cuts_to_master = true,
        fraction_of_benders_cuts_to_master = 1.0,
        reuse_dcglp = true,
        lift = true,
    )

    for instance_index in 1:5
        instance_name = "f25-c50-s64-r10-$instance_index"
        data = read_stochastic_uncapacitated_facility_location_problem(instance_name)
        @assert(
            haskey(reference_objectives, instance_name),
            "Missing SUFLP reference objective for $instance_name.",
        )
        reference_objective = reference_objectives[instance_name]

        oracle_specs = [
            (
                name = "UFLKnapsackOracle",
                param = UFLKnapsackOracleParam(
                    add_only_violated_cuts = true,
                ),
            ),
        ]

        @testset "Instance: $instance_name" begin
            for (spec_index, spec) in enumerate(oracle_specs)
                @testset "$(spec.name)" begin
                    Random.seed!(10_000 * instance_index + spec_index)
                    master = Master(
                        data;
                        model = update_knapsack_master_model!,
                        optimizer = mip_optimizer,
                    )

                    split_oracles = [
                        build_local_split_oracle(
                            data,
                            master,
                            scenario,
                            spec.param,
                            split_param,
                        )
                        for scenario in 1:data.n_scenarios
                    ]
                    oracle = SeparableOracle(master, split_oracles)
                    env = BendersSeq(master, oracle; param = benders_param)

                    @info(
                        "solving SUFLP separable local-split experiment",
                        instance = instance_name,
                        oracle = spec.name,
                    )
                    elapsed = @elapsed solve!(env)
                    disjunctive_cut_count = sum(
                        length(child.disjunctive_cuts) for child in oracle.oracles
                    )
                    absolute_error = abs(env.obj_value - reference_objective)
                    @info(
                        "SUFLP separable local-split result",
                        instance = instance_name,
                        oracle = spec.name,
                        termination_status = env.termination_status,
                        objective_value = env.obj_value,
                        reference_objective = reference_objective,
                        absolute_error = absolute_error,
                        disjunctive_cut_count = disjunctive_cut_count,
                        elapsed_seconds = elapsed,
                    )

                    @test env.termination_status == Optimal()
                    @test isapprox(reference_objective, env.obj_value; atol = 1.0e-5)
                    @test all(
                        child.dim_auxiliary == data.n_customers
                        for child in oracle.oracles
                    )
                    @test all(
                        length(child.dcglp[:st]) == data.n_customers
                        for child in oracle.oracles
                    )
                    @test all(
                        length(cut.a_t) == data.n_customers
                        for child in oracle.oracles for cut in child.disjunctive_cuts
                    )
                end
            end
        end
    end
end
