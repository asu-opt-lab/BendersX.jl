using BendersX
using CSV
using DataFrames
using Test
using JuMP
using CPLEX
include(normpath(joinpath(@__DIR__, "..", "solver_defaults.jl")))

@testset verbose = true "CFLP Callback Benders Tests" begin
    reference_path = normpath(joinpath(@__DIR__, "..", "reference_objectives", "cflp.csv"))
    reference_df = DataFrame(CSV.File(reference_path))
    @assert nrow(reference_df) == length(unique(reference_df.instance_name)) "Duplicate CFLP reference objectives found in $(reference_path)"
    reference_objectives = Dict(String(row.instance_name) => Float64(row.objective_value) for row in eachrow(reference_df))
    instances = setdiff(1:71, [67])

    # GBC-enabled subproblem customization (y[i,j] <= x[i] via GBC)
    function update_sub_gbc_model!(model::Model, data::CFLPData, scen_idx::Int; x)
        optimizer = optimizer_with_attributes(CPLEX.Optimizer, "CPXPARAM_Threads" => 7, "CPX_PARAM_EPRHS" => 1e-9, "CPX_PARAM_EPOPT" => 1e-9, "CPX_PARAM_NUMERICALEMPHASIS" => 1, MOI.Silent() => true)
        set_optimizer(model, optimizer)

        I, J = data.n_facilities, data.n_customers
        @variable(model, y[1:I, 1:J] >= 0)

        cost_demands = data.costs .* data.demands'
        @objective(model, Min, sum(cost_demands .* y))

        @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
        @constraint(model, capacity[i in 1:I], sum(data.demands[:] .* y[i,:]) <= data.capacities[i] * x[i])

        # Return GBC tuple: y[i,j] <= x[i]
        gbc_lhs = vec(y)
        gbc_rhs = [x[i] for j in 1:J for i in 1:I]  # j outer, i inner to match vec(y)
        gbc_sense = fill(UpperBound, I*J)
        return gbc_lhs, gbc_rhs, gbc_sense
    end

    for i in instances
        @testset "Instance: p$i" begin
            @info "Testing CFLP easy instance $i"

            # Load problem data
            data = read_cflp_benchmark_data("p$i")

            # BnB parameters
            benders_param = BendersBnBParam(;
                            time_limit = 200.0,
                            gap_tolerance = 1e-6,
                            verbose = false
                        )

            instance_name = "p$i"
            @assert haskey(reference_objectives, instance_name) "Missing CFLP reference objective for $(instance_name) in $(reference_path)"
            mip_opt_val = reference_objectives[instance_name]

            @testset "Classic oracle" begin
                @testset "NoSeq" begin
                    @info "solving CFLP p$i - classical oracle - no seq..."
                    # This setting can use default initializer
                    master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                    oracle = ClassicalOracle(data, master; model = update_sub_model!, optimizer = optimizer)

                    # preprocessing = NoPreprocessing()
                    # lazy_callback = LazyCallback(oracle)
                    # user_callback = NoUserCallback()
                    # env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)

                    env = BendersBnB(master, oracle; param = benders_param)
                    log = solve!(env)
                    @test env.termination_status == Optimal()
                    @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                end

                @testset "Seq" begin
                    @info "solving CFLP p$i - classical oracle - seq..."
                    master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                    oracle = ClassicalOracle(data, master; model = update_sub_model!, optimizer = optimizer)

                    preprocessing_seq_type = BendersSeq
                    preprocessing_seq_param = BendersSeqParam(;
                                time_limit = 200.0,
                                gap_tolerance = 1e-9,
                                verbose = false
                            )

                    preprocessing = LPRelaxationPreprocessing(oracle; seq_env_type = preprocessing_seq_type, param = preprocessing_seq_param)
                    lazy_callback = LazyCallback(oracle)
                    user_callback = NoUserCallback()

                    env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                    log = solve!(env)
                    @test env.termination_status == Optimal()
                    @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                end

                @testset "SeqInOut" begin
                    @info "solving CFLP p$i - classical oracle - seqinout..."
                    master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                    oracle = ClassicalOracle(data, master; model = update_sub_model!, optimizer = optimizer)

                    preprocessing_seq_type = BendersSeqInOut
                    preprocessing_seq_param = BendersSeqInOutParam(
                                time_limit = 300.0,
                                gap_tolerance = 1e-9,
                                stabilizing_x = ones(data.n_facilities),
                                α = 0.9,
                                λ = 0.1,
                                verbose = false
                            )

                    preprocessing = LPRelaxationPreprocessing(oracle; seq_env_type = preprocessing_seq_type, param = preprocessing_seq_param)
                    lazy_callback = LazyCallback(oracle)
                    user_callback = NoUserCallback()

                    env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                    log = solve!(env)
                    @test env.termination_status == Optimal()
                    @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                end
            end

            @testset "Knapsack oracle" begin
                @testset "NoSeq" begin
                    @info "solving CFLP p$i - knapsack oracle - no seq..."
                    master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                    oracle = CFLKnapsackOracle(data, master; model = update_sub_model!, optimizer = optimizer)

                    preprocessing = NoPreprocessing()
                    lazy_callback = LazyCallback(oracle)
                    user_callback = NoUserCallback()

                    env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                    log = solve!(env)
                    @test env.termination_status == Optimal()
                    @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                end

                @testset "Seq" begin
                    @info "solving CFLP p$i - knapsack oracle - seq..."
                    master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                    oracle = CFLKnapsackOracle(data, master; model = update_sub_model!, optimizer = optimizer)

                    preprocessing_seq_type = BendersSeq
                    preprocessing_seq_param = BendersSeqParam(;
                                time_limit = 200.0,
                                gap_tolerance = 1e-9,
                                verbose = false
                            )

                    preprocessing = LPRelaxationPreprocessing(oracle; seq_env_type = preprocessing_seq_type, param = preprocessing_seq_param)
                    lazy_callback = LazyCallback(oracle)
                    user_callback = NoUserCallback()

                    env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                    log = solve!(env)
                    @test env.termination_status == Optimal()
                    @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                end

                @testset "SeqInOut" begin
                    @info "solving CFLP p$i - knapsack oracle - seqinout..."
                    master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                    oracle = CFLKnapsackOracle(data, master; model = update_sub_model!, optimizer = optimizer)

                    preprocessing_seq_type = BendersSeqInOut
                    preprocessing_seq_param = BendersSeqInOutParam(
                                time_limit = 300.0,
                                gap_tolerance = 1e-9,
                                stabilizing_x = ones(data.n_facilities),
                                α = 0.9,
                                λ = 0.1,
                                verbose = false
                            )

                    preprocessing = LPRelaxationPreprocessing(oracle; seq_env_type = preprocessing_seq_type, param = preprocessing_seq_param)
                    lazy_callback = LazyCallback(oracle)
                    user_callback = NoUserCallback()

                    env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                    log = solve!(env)
                    @test env.termination_status == Optimal()
                    @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                end
            end

            @testset "Unified oracle" begin
                @testset "NoSeq" begin
                    @info "solving CFLP p$i - unified oracle - no seq..."
                    master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                    oracle = UnifiedOracle(data, master; model = update_sub_model!, optimizer = optimizer)
                    env = BendersBnB(master, oracle; param = benders_param)
                    log = solve!(env)
                    @test env.termination_status == Optimal()
                    @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                end

                @testset "Seq" begin
                    @info "solving CFLP p$i - unified oracle - seq..."
                    master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                    oracle = UnifiedOracle(data, master; model = update_sub_model!, optimizer = optimizer)

                    preprocessing_seq_type = BendersSeq
                    preprocessing_seq_param = BendersSeqParam(;
                                time_limit = 200.0,
                                gap_tolerance = 1e-9,
                                verbose = false
                            )

                    preprocessing = LPRelaxationPreprocessing(oracle; seq_env_type = preprocessing_seq_type, param = preprocessing_seq_param)
                    lazy_callback = LazyCallback(oracle)
                    user_callback = NoUserCallback()

                    env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                    log = solve!(env)
                    @test env.termination_status == Optimal()
                    @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                end

                @testset "SeqInOut" begin
                    @info "solving CFLP p$i - unified oracle - seqinout..."
                    master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                    oracle = UnifiedOracle(data, master; model = update_sub_model!, optimizer = optimizer)

                    preprocessing_seq_type = BendersSeqInOut
                    preprocessing_seq_param = BendersSeqInOutParam(
                                time_limit = 300.0,
                                gap_tolerance = 1e-9,
                                stabilizing_x = ones(data.n_facilities),
                                α = 0.9,
                                λ = 0.1,
                                verbose = false
                            )

                    preprocessing = LPRelaxationPreprocessing(oracle; seq_env_type = preprocessing_seq_type, param = preprocessing_seq_param)
                    lazy_callback = LazyCallback(oracle)
                    user_callback = NoUserCallback()

                    env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                    log = solve!(env)
                    @test env.termination_status == Optimal()
                    @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                end
            end

            @testset "Classic oracle with GBC" begin
                @testset "NoSeq" begin
                    @info "solving CFLP p$i - classical oracle with GBC - no seq..."
                    master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                    oracle = ClassicalOracle(data, master; model = update_sub_gbc_model!, optimizer = optimizer)
                    env = BendersBnB(master, oracle; param = benders_param)
                    log = solve!(env)
                    @test env.termination_status == Optimal()
                    @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                end

                @testset "Seq" begin
                    @info "solving CFLP p$i - classical oracle with GBC - seq..."
                    master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                    oracle = ClassicalOracle(data, master; model = update_sub_gbc_model!, optimizer = optimizer)

                    preprocessing_seq_type = BendersSeq
                    preprocessing_seq_param = BendersSeqParam(;
                                time_limit = 200.0,
                                gap_tolerance = 1e-9,
                                verbose = false
                            )

                    preprocessing = LPRelaxationPreprocessing(oracle; seq_env_type = preprocessing_seq_type, param = preprocessing_seq_param)
                    lazy_callback = LazyCallback(oracle)
                    user_callback = NoUserCallback()

                    env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                    log = solve!(env)
                    @test env.termination_status == Optimal()
                    @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                end

                @testset "SeqInOut" begin
                    @info "solving CFLP p$i - classical oracle with GBC - seqinout..."
                    master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                    oracle = ClassicalOracle(data, master; model = update_sub_gbc_model!, optimizer = optimizer)

                    preprocessing_seq_type = BendersSeqInOut
                    preprocessing_seq_param = BendersSeqInOutParam(
                                time_limit = 300.0,
                                gap_tolerance = 1e-9,
                                stabilizing_x = ones(data.n_facilities),
                                α = 0.9,
                                λ = 0.1,
                                verbose = false
                            )

                    preprocessing = LPRelaxationPreprocessing(oracle; seq_env_type = preprocessing_seq_type, param = preprocessing_seq_param)
                    lazy_callback = LazyCallback(oracle)
                    user_callback = NoUserCallback()

                    env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                    log = solve!(env)
                    @test env.termination_status == Optimal()
                    @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                end
            end

            @testset "Pareto oracle" begin
                @testset "NoSeq" begin
                    @info "solving CFLP p$i - pareto oracle - no seq..."
                    master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                    pareto_param = ParetoOracleParam(ones(data.n_facilities))
                    oracle = ParetoOracle(data, master, pareto_param; model = update_sub_model!, optimizer = optimizer)
                    env = BendersBnB(master, oracle; param = benders_param)
                    log = solve!(env)
                    @test env.termination_status == Optimal()
                    @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                end

                @testset "Seq" begin
                    @info "solving CFLP p$i - pareto oracle - seq..."
                    master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                    pareto_param = ParetoOracleParam(ones(data.n_facilities))
                    oracle = ParetoOracle(data, master, pareto_param; model = update_sub_model!, optimizer = optimizer)

                    preprocessing_seq_type = BendersSeq
                    preprocessing_seq_param = BendersSeqParam(;
                                time_limit = 200.0,
                                gap_tolerance = 1e-9,
                                verbose = false
                            )

                    preprocessing = LPRelaxationPreprocessing(oracle; seq_env_type = preprocessing_seq_type, param = preprocessing_seq_param)
                    lazy_callback = LazyCallback(oracle)
                    user_callback = NoUserCallback()

                    env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                    log = solve!(env)
                    @test env.termination_status == Optimal()
                    @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                end

                @testset "SeqInOut" begin
                    @info "solving CFLP p$i - pareto oracle - seqinout..."
                    master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                    pareto_param = ParetoOracleParam(ones(data.n_facilities))
                    oracle = ParetoOracle(data, master, pareto_param; model = update_sub_model!, optimizer = optimizer)

                    preprocessing_seq_type = BendersSeqInOut
                    preprocessing_seq_param = BendersSeqInOutParam(
                                time_limit = 300.0,
                                gap_tolerance = 1e-9,
                                stabilizing_x = ones(data.n_facilities),
                                α = 0.9,
                                λ = 0.1,
                                verbose = false
                            )

                    preprocessing = LPRelaxationPreprocessing(oracle; seq_env_type = preprocessing_seq_type, param = preprocessing_seq_param)
                    lazy_callback = LazyCallback(oracle)
                    user_callback = NoUserCallback()

                    env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                    log = solve!(env)
                    @test env.termination_status == Optimal()
                    @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                end
            end

            @testset "Knapsack oracle with GBC" begin
                @testset "NoSeq" begin
                    @info "solving CFLP p$i - knapsack oracle with GBC - no seq..."
                    master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                    oracle = CFLKnapsackOracle(data, master; model = update_sub_gbc_model!, optimizer = optimizer)
                    preprocessing = NoPreprocessing()
                    lazy_callback = LazyCallback(oracle)
                    user_callback = NoUserCallback()
                    env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                    log = solve!(env)
                    @test env.termination_status == Optimal()
                    @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                end

                @testset "Seq" begin
                    @info "solving CFLP p$i - knapsack oracle with GBC - seq..."
                    master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                    oracle = CFLKnapsackOracle(data, master; model = update_sub_gbc_model!, optimizer = optimizer)

                    preprocessing_seq_type = BendersSeq
                    preprocessing_seq_param = BendersSeqParam(;
                                time_limit = 200.0,
                                gap_tolerance = 1e-9,
                                verbose = false
                            )

                    preprocessing = LPRelaxationPreprocessing(oracle; seq_env_type = preprocessing_seq_type, param = preprocessing_seq_param)
                    lazy_callback = LazyCallback(oracle)
                    user_callback = NoUserCallback()

                    env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                    log = solve!(env)
                    @test env.termination_status == Optimal()
                    @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                end

                @testset "SeqInOut" begin
                    @info "solving CFLP p$i - knapsack oracle with GBC - seqinout..."
                    master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                    oracle = CFLKnapsackOracle(data, master; model = update_sub_gbc_model!, optimizer = optimizer)

                    preprocessing_seq_type = BendersSeqInOut
                    preprocessing_seq_param = BendersSeqInOutParam(
                                time_limit = 300.0,
                                gap_tolerance = 1e-9,
                                stabilizing_x = ones(data.n_facilities),
                                α = 0.9,
                                λ = 0.1,
                                verbose = false
                            )

                    preprocessing = LPRelaxationPreprocessing(oracle; seq_env_type = preprocessing_seq_type, param = preprocessing_seq_param)
                    lazy_callback = LazyCallback(oracle)
                    user_callback = NoUserCallback()

                    env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                    log = solve!(env)
                    @test env.termination_status == Optimal()
                    @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                end
            end
        end
    end
end
