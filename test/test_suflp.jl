using Test
using BendersX
using HiGHS
using JuMP
using MathOptInterface

const SUFLP_TEST_MOI = MathOptInterface

function suflp_test_optimizer()
    return optimizer_with_attributes(
        HiGHS.Optimizer,
        SUFLP_TEST_MOI.Silent() => true,
    )
end

function suflp_test_data()
    return SUFLPData(
        3,
        2,
        2,
        [[1.0, 3.0], [4.0, 1.0]],
        [0.25, 0.75],
        [4.0, 5.0, 9.0],
        [1.0 5.0; 4.0 1.0; 2.0 2.0],
    )
end

function suflp_test_split_param()
    return SplitOracleParam(;
        dcglp_param = DcglpParam(;
            optimizer = suflp_test_optimizer(),
            time_limit = 20.0,
            gap_tolerance = 1.0e-6,
            halt_limit = 3,
            iter_limit = 20,
            verbose = false,
        ),
        split_index_selection_rule = MostFractional(),
        disjunctive_cut_append_rule = AllDisjunctiveCuts(),
        strengthened = false,
        lift = false,
    )
end

function solve_suflp_extensive(data::SUFLPData)
    model = Model(suflp_test_optimizer())
    I, J, S = data.n_facilities, data.n_customers, data.n_scenarios
    @variable(model, x[1:I], Bin)
    @variable(model, y[1:I, 1:J, 1:S] >= 0)
    @constraint(model, sum(x) >= 1)
    @constraint(model, [j in 1:J, s in 1:S], sum(y[:, j, s]) == 1)
    @constraint(model, [i in 1:I, j in 1:J, s in 1:S], y[i, j, s] <= x[i])
    @objective(
        model,
        Min,
        data.fixed_costs' * x + sum(
            data.probabilities[s] * data.costs[i, j] *
            data.demands[s][j] * y[i, j, s]
            for i in 1:I, j in 1:J, s in 1:S
        ),
    )
    optimize!(model)
    @test termination_status(model) == SUFLP_TEST_MOI.OPTIMAL
    return objective_value(model)
end

function solve_suflp_benders(data::SUFLPData, master_model, oracle)
    master = Master(
        data;
        model = master_model,
        optimizer = suflp_test_optimizer(),
    )
    separable = oracle(data, master)
    env = BendersSeq(
        master,
        separable;
        param = BendersSeqParam(
            time_limit = 30.0,
            gap_tolerance = 1.0e-7,
            verbose = false,
        ),
    )
    solve!(env)
    return env, master, separable
end

@testset "Stochastic uncapacitated facility location" begin
    @testset "constructs data and supports deterministic/stochastic conversion" begin
        data = suflp_test_data()
        @test data.probabilities == [0.25, 0.75]

        deterministic = UFLPData(
            2,
            2,
            [1.0, 2.0],
            [3.0, 4.0],
            [1.0 2.0; 3.0 4.0],
        )
        stochastic = SUFLPData(deterministic)
        @test stochastic.n_scenarios == 1
        @test stochastic.demands == [[1.0, 2.0]]
        @test stochastic.probabilities == [1.0]

        capacitated = SCFLPData(
            2,
            2,
            2,
            [10.0, 12.0],
            [[1.0, 2.0], [3.0, 4.0]],
            [3.0, 4.0],
            [1.0 2.0; 3.0 4.0],
        )
        uncapacitated = SUFLPData(capacitated; probabilities = [0.4, 0.6])
        @test uncapacitated.demands == capacitated.demands
        @test uncapacitated.probabilities == [0.4, 0.6]

        mktempdir() do directory
            open(joinpath(directory, "tiny.json"), "w") do io
                write(
                    io,
                    """
                    {
                        "n_facilities": 2,
                        "n_customers": 2,
                        "n_scenarios": 2,
                        "demands": [[1.0, 2.0], [3.0, 4.0]],
                        "probabilities": [0.3, 0.7],
                        "fixed_costs": [5.0, 6.0],
                        "costs": [[1.0, 2.0], [3.0, 4.0]]
                    }
                    """,
                )
            end
            parsed = read_stochastic_uncapacitated_facility_location_problem(
                "tiny";
                filepath = directory,
            )
            @test parsed.probabilities == [0.3, 0.7]
            @test parsed.costs == [1.0 2.0; 3.0 4.0]
        end
    end

    @testset "keeps each scenario's customer block contiguous" begin
        data = suflp_test_data()
        master = Master(
            data;
            model = update_knapsack_master_model!,
            optimizer = suflp_test_optimizer(),
        )
        oracle = SeparableOracle(
            data,
            master,
            UFLKnapsackOracle,
            data.n_scenarios;
            sub_oracle_param = UFLKnapsackOracleParam(),
            optimizer = suflp_test_optimizer(),
        )

        @test master.dim_t == data.n_customers * data.n_scenarios
        @test master.c_t == [0.25, 0.25, 0.75, 0.75]
        @test oracle.auxiliary_ranges == [1:2, 3:4]
        @test oracle.dim_auxiliary == 4
        @test BendersX.auxiliary_dimension.(oracle.oracles) == [2, 2]

        is_in_L, cuts, objectives = BendersX.generate_cuts(
            oracle,
            [1.0, 0.0, 0.0],
            zeros(4),
        )
        @test !is_in_L
        @test objectives == [1.0, 15.0, 4.0, 5.0]
        @test collect.(getfield.(cuts, :a_t)) == [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
        ]

        split_children = [
            SplitOracle(
                master,
                (
                    UFLKnapsackOracle(data; scen_idx = scenario),
                    UFLKnapsackOracle(data; scen_idx = scenario),
                );
                normalization = LpDistanceNormalization(1.0),
                param = suflp_test_split_param(),
            ) for scenario in 1:data.n_scenarios
        ]
        split_oracle = SeparableOracle(master, split_children)
        @test split_oracle.dim_auxiliary == 4
        @test split_oracle.auxiliary_ranges == [1:2, 3:4]
        @test BendersX.auxiliary_dimension.(split_children) == [2, 2]
        @test all(length(child.dcglp[:st]) == 2 for child in split_children)
    end

    @testset "Benders formulations match the extensive form" begin
        data = suflp_test_data()
        extensive_objective = solve_suflp_extensive(data)
        @test isapprox(extensive_objective, 13.75; atol = 1.0e-8)

        classical_env, classical_master, classical_oracle = solve_suflp_benders(
            data,
            update_master_model!,
            (data, master) -> SeparableOracle(
                data,
                master,
                ClassicalOracle,
                data.n_scenarios;
                model = update_sub_model!,
                optimizer = suflp_test_optimizer(),
            ),
        )
        @test classical_env.termination_status == Optimal()
        @test classical_master.dim_t == data.n_scenarios
        @test classical_oracle.auxiliary_ranges == [1:1, 2:2]
        @test isapprox(classical_env.obj_value, extensive_objective; atol = 1.0e-6)

        knapsack_env, knapsack_master, knapsack_oracle = solve_suflp_benders(
            data,
            update_knapsack_master_model!,
            (data, master) -> SeparableOracle(
                data,
                master,
                UFLKnapsackOracle,
                data.n_scenarios;
                sub_oracle_param = UFLKnapsackOracleParam(
                    add_only_violated_cuts = true,
                ),
                optimizer = suflp_test_optimizer(),
            ),
        )
        @test knapsack_env.termination_status == Optimal()
        @test knapsack_master.dim_t == data.n_customers * data.n_scenarios
        @test knapsack_oracle.auxiliary_ranges == [1:2, 3:4]
        @test isapprox(knapsack_env.obj_value, extensive_objective; atol = 1.0e-6)
    end
end
