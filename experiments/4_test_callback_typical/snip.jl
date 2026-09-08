using BendersX
using CSV
using DataFrames
using Test
using JuMP
include(normpath(joinpath(@__DIR__, "..", "solver_defaults.jl")))

@testset verbose = true "SNIP Sequential Benders Tests" begin
    reference_path = normpath(joinpath(@__DIR__, "..", "reference_objectives", "snip.csv"))
    reference_df = DataFrame(CSV.File(reference_path))
    @assert nrow(reference_df) == length(unique(reference_df.instance_name)) "Duplicate SNIP reference objectives found in $(reference_path)"
    reference_objectives = Dict(String(row.instance_name) => Float64(row.objective_value) for row in eachrow(reference_df))
    for instance in [0], snipno in [0], budget in [30.0]
        @testset "instance $instance; snipno $snipno budget $budget" begin
            data = read_snip_data(instance, snipno, budget)
            
            # BnB parameters
            benders_param = BendersBnBParam(;
                            time_limit = 200.0,
                            verbose = false
                        )
                
            instance_name = "instance=$(instance);snipno=$(snipno);budget=$(budget)"
            @assert haskey(reference_objectives, instance_name) "Missing SNIP reference objective for $(instance_name) in $(reference_path)"
            mip_opt_val = reference_objectives[instance_name]

            @testset "Classic oracle" begin
                @testset "NoSeq" begin
                    @info "solving SNIP instance$instance snipno $snipno budget $budget - classical oracle - no seq..."
                    master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                    oracle = SeparableOracle(data, master, ClassicalOracle, data.num_scenarios; model = update_sub_model!, optimizer = optimizer)

                    preprocessing = NoPreprocessing()
                    lazy_callback = LazyCallback(oracle)
                    user_callback = NoUserCallback()

                    env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                    log = solve!(env)
                    @test env.termination_status == Optimal()
                    @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                end

                @testset "Seq" begin
                    @info "solving SNIP instance$instance snipno $snipno budget $budget - classical oracle - seq..."
                    master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                    oracle = SeparableOracle(data, master, ClassicalOracle, data.num_scenarios; model = update_sub_model!, optimizer = optimizer)

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
                    @info "solving SNIP instance$instance snipno $snipno budget $budget - classical oracle - seqinout..."
                    master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                    oracle = SeparableOracle(data, master, ClassicalOracle, data.num_scenarios; model = update_sub_model!, optimizer = optimizer)

                    preprocessing_seq_type = BendersSeqInOut
                    preprocessing_seq_param = BendersSeqInOutParam(
                                time_limit = 300.0,
                                gap_tolerance = 1e-9,
                                stabilizing_x = ones(length(data.D)),
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
                    @info "solving SNIP instance$instance snipno $snipno budget $budget - pareto oracle - no seq..."
                    master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                    pareto_param = ParetoOracleParam(ones(length(data.D)))
                    oracle = SeparableOracle(data, master, ParetoOracle, data.num_scenarios; model = update_sub_model!, sub_oracle_param = pareto_param, optimizer = optimizer)

                    preprocessing = NoPreprocessing()
                    lazy_callback = LazyCallback(oracle)
                    user_callback = NoUserCallback()

                    env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                    log = solve!(env)
                    @test env.termination_status == Optimal()
                    @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                end

                @testset "Seq" begin
                    @info "solving SNIP instance$instance snipno $snipno budget $budget - pareto oracle - seq..."
                    master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                    pareto_param = ParetoOracleParam(ones(length(data.D)))
                    oracle = SeparableOracle(data, master, ParetoOracle, data.num_scenarios; model = update_sub_model!, sub_oracle_param = pareto_param, optimizer = optimizer)

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
                    @info "solving SNIP instance$instance snipno $snipno budget $budget - pareto oracle - seqinout..."
                    master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                    pareto_param = ParetoOracleParam(ones(length(data.D)))
                    oracle = SeparableOracle(data, master, ParetoOracle, data.num_scenarios; model = update_sub_model!, sub_oracle_param = pareto_param, optimizer = optimizer)

                    preprocessing_seq_type = BendersSeqInOut
                    preprocessing_seq_param = BendersSeqInOutParam(
                                time_limit = 300.0,
                                gap_tolerance = 1e-9,
                                stabilizing_x = ones(length(data.D)),
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
                    @info "solving SNIP instance$instance snipno $snipno budget $budget - unified oracle - no seq..."
                    master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                    oracle = SeparableOracle(data, master, UnifiedOracle, data.num_scenarios; model = update_sub_model!, sub_oracle_param = UnifiedOracleParam(), optimizer = optimizer)

                    preprocessing = NoPreprocessing()
                    lazy_callback = LazyCallback(oracle)
                    user_callback = NoUserCallback()

                    env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                    log = solve!(env)
                    @test env.termination_status == Optimal()
                    @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                end

                @testset "Seq" begin
                    @info "solving SNIP instance$instance snipno $snipno budget $budget - unified oracle - seq..."
                    master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                    oracle = SeparableOracle(data, master, UnifiedOracle, data.num_scenarios; model = update_sub_model!, sub_oracle_param = UnifiedOracleParam(), optimizer = optimizer)

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
                    @info "solving SNIP instance$instance snipno $snipno budget $budget - unified oracle - seqinout..."
                    master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                    oracle = SeparableOracle(data, master, UnifiedOracle, data.num_scenarios; model = update_sub_model!, sub_oracle_param = UnifiedOracleParam(), optimizer = optimizer)

                    preprocessing_seq_type = BendersSeqInOut
                    preprocessing_seq_param = BendersSeqInOutParam(
                                time_limit = 300.0,
                                gap_tolerance = 1e-9,
                                stabilizing_x = ones(length(data.D)),
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
