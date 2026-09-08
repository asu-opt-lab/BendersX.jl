using BendersX
using CSV
using CPLEX
using DataFrames
using JuMP
using Random
using Test

include(normpath(joinpath(@__DIR__, "..", "solver_defaults.jl")))

function update_separable_split_sub_gbc_model!(
    model::Model,
    data::SCFLPData,
    scen_idx::Int;
    x,
)
    subproblem_optimizer = optimizer_with_attributes(
        CPLEX.Optimizer,
        "CPXPARAM_Threads" => 7,
        "CPX_PARAM_EPRHS" => 1.0e-9,
        "CPX_PARAM_EPOPT" => 1.0e-9,
        "CPX_PARAM_NUMERICALEMPHASIS" => 1,
        MOI.Silent() => true,
    )
    set_optimizer(model, subproblem_optimizer)

    I, J = data.n_facilities, data.n_customers
    @variable(model, y[1:I, 1:J] >= 0)
    cost_demands = data.costs .* data.demands[scen_idx]'
    @objective(model, Min, sum(cost_demands .* y))
    @constraint(model, demand[j in 1:J], sum(y[:, j]) == 1)
    @constraint(
        model,
        capacity[i in 1:I],
        sum(data.demands[scen_idx][j] * y[i, j] for j in 1:J) <=
        data.capacities[i] * x[i],
    )

    gbc_lhs = vec(y)
    gbc_rhs = [x[i] for j in 1:J for i in 1:I]
    gbc_sense = fill(UpperBound, I * J)
    return gbc_lhs, gbc_rhs, gbc_sense
end

function build_local_split_oracle(
    data::SCFLPData,
    master::Master,
    scenario::Int,
    oracle_type::Type{T},
    model,
    oracle_param::BendersX.AbstractOracleParam,
    split_param::SplitOracleParam,
) where {T<:BendersX.AbstractTypicalOracle}
    kappa = oracle_type(
        data,
        master;
        model = model,
        scen_idx = scenario,
        param = deepcopy(oracle_param),
        optimizer = optimizer,
    )
    nu = oracle_type(
        data,
        master;
        model = model,
        scen_idx = scenario,
        param = deepcopy(oracle_param),
        optimizer = optimizer,
    )

    return SplitOracle(
        master,
        (kappa, nu);
        dim_t = 1,
        param = deepcopy(split_param),
    )
end

@testset verbose = true "SCFLP Sequential Separable SplitOracle Tests" begin
    reference_path = normpath(
        joinpath(@__DIR__, "..", "reference_objectives", "scflp.csv"),
    )
    reference_df = DataFrame(CSV.File(reference_path))
    @assert(
        nrow(reference_df) == length(unique(reference_df.instance_name)),
        "Duplicate SCFLP reference objectives found in $reference_path.",
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
        normalization = LpDistanceNormalization(1.0),
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
        data = read_stochastic_capacited_facility_location_problem(instance_name)
        @assert(
            haskey(reference_objectives, instance_name),
            "Missing SCFLP reference objective for $instance_name.",
        )
        reference_objective = reference_objectives[instance_name]

        oracle_specs = [
            (
                name = "ClassicalOracle",
                oracle_type = ClassicalOracle,
                model = update_sub_model!,
                param = BasicOracleParam(),
            ),
            (
                name = "CFLKnapsackOracle",
                oracle_type = CFLKnapsackOracle,
                model = update_sub_model!,
                param = CFLKnapsackOracleParam(),
            ),
            (
                name = "UnifiedOracle",
                oracle_type = UnifiedOracle,
                model = update_sub_model!,
                param = UnifiedOracleParam(),
            ),
            (
                name = "ParetoOracle",
                oracle_type = ParetoOracle,
                model = update_sub_model!,
                param = ParetoOracleParam(ones(data.n_facilities)),
            ),
            (
                name = "ClassicalOracle with GBC",
                oracle_type = ClassicalOracle,
                model = update_separable_split_sub_gbc_model!,
                param = BasicOracleParam(),
            ),
            (
                name = "CFLKnapsackOracle with GBC",
                oracle_type = CFLKnapsackOracle,
                model = update_separable_split_sub_gbc_model!,
                param = CFLKnapsackOracleParam(),
            ),
        ]

        @testset "Instance: $instance_name" begin
            for (spec_index, spec) in enumerate(oracle_specs)
                @testset "$(spec.name)" begin
                    # Make RandomFractional reproducible for every configuration.
                    Random.seed!(10_000 * instance_index + spec_index)
                    master = Master(
                        data;
                        model = update_master_model!,
                        optimizer = mip_optimizer,
                    )

                    # New composition: each scenario has its own one-dimensional
                    # SplitOracle/DCGLP; SeparableOracle embeds local cuts at t[j].
                    split_oracles = [
                        build_local_split_oracle(
                            data,
                            master,
                            scenario,
                            spec.oracle_type,
                            spec.model,
                            spec.param,
                            split_param,
                        )
                        for scenario in 1:data.n_scenarios
                    ]
                    oracle = SeparableOracle(master, split_oracles)
                    env = BendersSeq(master, oracle; param = benders_param)

                    @info(
                        "solving SCFLP separable local-split experiment",
                        instance = instance_name,
                        oracle = spec.name,
                    )
                    elapsed = @elapsed solve!(env)
                    disjunctive_cut_count = sum(
                        length(child.disjunctive_cuts) for child in oracle.oracles
                    )
                    absolute_error = abs(env.obj_value - reference_objective)
                    @info(
                        "SCFLP separable local-split result",
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
                    @test all(child.dim_t == 1 for child in oracle.oracles)
                    @test all(length(child.dcglp[:st]) == 1 for child in oracle.oracles)
                    @test all(
                        length(cut.a_t) == 1
                        for child in oracle.oracles for cut in child.disjunctive_cuts
                    )
                end
            end
        end
    end
end
