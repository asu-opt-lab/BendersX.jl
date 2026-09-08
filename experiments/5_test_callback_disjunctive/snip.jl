using BendersX
using CSV
using DataFrames
using Test
using JuMP
using CPLEX
include(normpath(joinpath(@__DIR__, "..", "solver_defaults.jl")))

@testset verbose = true "SNIP Sequential Benders Tests" begin
    reference_path = normpath(joinpath(@__DIR__, "..", "reference_objectives", "snip.csv"))
    reference_df = DataFrame(CSV.File(reference_path))
    @assert nrow(reference_df) == length(unique(reference_df.instance_name)) "Duplicate SNIP reference objectives found in $(reference_path)"
    reference_objectives = Dict(String(row.instance_name) => Float64(row.objective_value) for row in eachrow(reference_df))
    for instance in [0], snipno in [0], budget in [30.0]
        @testset "instance $instance; snipno $snipno budget $budget" begin
            # Load problem data
            data = read_snip_data(instance, snipno, budget)

            # Algorithm parameters
            benders_param = BendersBnBParam(;
                                            time_limit = 3600.0,
                                            gap_tolerance = 1e-6,
                                            verbose = false
                                            )
            dcglp_optimizer = optimizer_with_attributes(CPLEX.Optimizer, "CPXPARAM_Threads" => 7, "CPX_PARAM_EPRHS" => 1e-9, "CPX_PARAM_NUMERICALEMPHASIS" => 1, "CPX_PARAM_EPOPT" => 1e-9, MOI.Silent() => true)
            dcglp_param = DcglpParam(; optimizer = dcglp_optimizer,
                                    time_limit = 200.0,
                                    gap_tolerance = 1e-3,
                                    halt_limit = 3,
                                    iter_limit = 15,
                                    verbose = false
                                    )

            user_cb_param = UserCallbackParam(frequency=100)

            instance_name = "instance=$(instance);snipno=$(snipno);budget=$(budget)"
            @assert haskey(reference_objectives, instance_name) "Missing SNIP reference objective for $(instance_name) in $(reference_path)"
            mip_opt_val = reference_objectives[instance_name]
            
            @testset "Classic oracle" begin
                # for strengthened in [true; false], add_benders_cuts_to_master in [true; false; 2], reuse_dcglp in [true; false], p in [1.0; Inf], lift in [true; false], disjunctive_cut_append_rule in [NoDisjunctiveCuts(); AllDisjunctiveCuts(); DisjunctiveCutsSmallerIndices()]
                for strengthened in [true], add_benders_cuts_to_master in [true], reuse_dcglp in [true], p in [Inf], lift in [true], disjunctive_cut_append_rule in [AllDisjunctiveCuts()]
                    @testset "strgthnd $strengthened; benders2master $add_benders_cuts_to_master; reuse $reuse_dcglp; p $p; lift $lift; dcut_append $disjunctive_cut_append_rule" begin

                        oracle_param = SplitOracleParam(; normalization = LpDistanceNormalization(p), dcglp_param = dcglp_param,
                                                            split_index_selection_rule = LargestFractional(),
                                                            disjunctive_cut_append_rule = disjunctive_cut_append_rule,
                                                            strengthened = strengthened,
                                                            add_benders_cuts_to_master = add_benders_cuts_to_master,
                                                            fraction_of_benders_cuts_to_master = 1.0,
                                                            reuse_dcglp = reuse_dcglp,
                                                            lift = lift)

                        @testset "NoSeq" begin
                            @info "solving SNIP instance$instance snipno $snipno budget $budget - disjunctive oracle/classical/no seq - strgthnd $strengthened; benders2master $add_benders_cuts_to_master reuse $reuse_dcglp lift $lift p $p dcut_append $disjunctive_cut_append_rule"
                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            lazy_oracle = SeparableOracle(data, master, ClassicalOracle, data.num_scenarios; model = update_sub_model!, optimizer = optimizer)
                            typical_oracle_kappa = SeparableOracle(data, master, ClassicalOracle, data.num_scenarios; model = update_sub_model!, optimizer = optimizer)
                            typical_oracle_nu = SeparableOracle(data, master, ClassicalOracle, data.num_scenarios; model = update_sub_model!, optimizer = optimizer)
                            typical_oracles = [typical_oracle_kappa; typical_oracle_nu]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); param = oracle_param)

                            preprocessing = NoPreprocessing()
                            lazy_callback = LazyCallback(lazy_oracle)
                            user_callback = UserCallback(disjunctive_oracle; param=user_cb_param)

                            env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                            log = solve!(env)
                            @test env.termination_status == Optimal()
                            @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                            if benders_param.verbose
                                @info "Disjunctive cuts added: $(length(env.user_callback.oracle.disjunctive_cuts))"
                                env.user_callback.oracle.param.add_benders_cuts_to_master != 0 && @info "Byproduct Benders cuts added: $(log.n_user_cuts[1] - length(env.user_callback.oracle.disjunctive_cuts))"
                            end
                        end

                        @testset "Seq" begin
                            @info "solving SNIP instance$instance snipno $snipno budget $budget - disjunctive oracle/classical/seq - strgthnd $strengthened; benders2master $add_benders_cuts_to_master reuse $reuse_dcglp lift $lift p $p dcut_append $disjunctive_cut_append_rule"
                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            lazy_oracle = SeparableOracle(data, master, ClassicalOracle, data.num_scenarios; model = update_sub_model!, optimizer = optimizer)
                            typical_oracle_kappa = SeparableOracle(data, master, ClassicalOracle, data.num_scenarios; model = update_sub_model!, optimizer = optimizer)
                            typical_oracle_nu = SeparableOracle(data, master, ClassicalOracle, data.num_scenarios; model = update_sub_model!, optimizer = optimizer)
                            typical_oracles = [typical_oracle_kappa; typical_oracle_nu]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); param = oracle_param)

                            preprocessing = LPRelaxationPreprocessing(lazy_oracle; seq_env_type = BendersSeq, param = BendersSeqParam(;time_limit=200.0, gap_tolerance=1e-9, verbose=false))
                            lazy_callback = LazyCallback(lazy_oracle)
                            user_callback = UserCallback(disjunctive_oracle; param=user_cb_param)

                            env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                            log = solve!(env)
                            @test env.termination_status == Optimal()
                            @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                            if benders_param.verbose
                                @info "Disjunctive cuts added: $(length(env.user_callback.oracle.disjunctive_cuts))"
                                env.user_callback.oracle.param.add_benders_cuts_to_master != 0 && @info "Byproduct Benders cuts added: $(log.n_user_cuts[1] - length(env.user_callback.oracle.disjunctive_cuts))"
                            end
                        end

                        @testset "SeqInOut" begin
                            @info "solving SNIP instance$instance snipno $snipno budget $budget - disjunctive oracle/classical/seqinout - strgthnd $strengthened; benders2master $add_benders_cuts_to_master reuse $reuse_dcglp lift $lift p $p dcut_append $disjunctive_cut_append_rule"
                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            lazy_oracle = SeparableOracle(data, master, ClassicalOracle, data.num_scenarios; model = update_sub_model!, optimizer = optimizer)
                            typical_oracle_kappa = SeparableOracle(data, master, ClassicalOracle, data.num_scenarios; model = update_sub_model!, optimizer = optimizer)
                            typical_oracle_nu = SeparableOracle(data, master, ClassicalOracle, data.num_scenarios; model = update_sub_model!, optimizer = optimizer)
                            typical_oracles = [typical_oracle_kappa; typical_oracle_nu]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); param = oracle_param)

                            preprocessing = LPRelaxationPreprocessing(lazy_oracle; seq_env_type = BendersSeqInOut, param = BendersSeqInOutParam(time_limit = 300.0, gap_tolerance = 1e-9, stabilizing_x = ones(length(data.D)), α = 0.9, λ = 0.1, verbose = false))
                            lazy_callback = LazyCallback(lazy_oracle)
                            user_callback = UserCallback(disjunctive_oracle; param=user_cb_param)

                            env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                            log = solve!(env)
                            @test env.termination_status == Optimal()
                            @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                            if benders_param.verbose
                                @info "Disjunctive cuts added: $(length(env.user_callback.oracle.disjunctive_cuts))"
                                env.user_callback.oracle.param.add_benders_cuts_to_master != 0 && @info "Byproduct Benders cuts added: $(log.n_user_cuts[1] - length(env.user_callback.oracle.disjunctive_cuts))"
                            end
                        end
                    end
                end
            end

            @testset "Pareto oracle" begin
                for strengthened in [true], add_benders_cuts_to_master in [true], reuse_dcglp in [true], p in [Inf], lift in [true], disjunctive_cut_append_rule in [AllDisjunctiveCuts()]
                    @testset "strgthnd $strengthened; benders2master $add_benders_cuts_to_master; reuse $reuse_dcglp; p $p; lift $lift; dcut_append $disjunctive_cut_append_rule" begin

                        oracle_param = SplitOracleParam(; normalization = LpDistanceNormalization(p), dcglp_param = dcglp_param,
                                                            split_index_selection_rule = LargestFractional(),
                                                            disjunctive_cut_append_rule = disjunctive_cut_append_rule,
                                                            strengthened = strengthened,
                                                            add_benders_cuts_to_master = add_benders_cuts_to_master,
                                                            fraction_of_benders_cuts_to_master = 1.0,
                                                            reuse_dcglp = reuse_dcglp,
                                                            lift = lift)

                        @testset "NoSeq" begin
                            @info "solving SNIP instance$instance snipno $snipno budget $budget - disjunctive oracle/pareto/no seq - strgthnd $strengthened; benders2master $add_benders_cuts_to_master reuse $reuse_dcglp lift $lift p $p dcut_append $disjunctive_cut_append_rule"
                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            pareto_param = ParetoOracleParam(ones(length(data.D)))
                            lazy_oracle = SeparableOracle(data, master, ParetoOracle, data.num_scenarios; model = update_sub_model!, sub_oracle_param = pareto_param, optimizer = optimizer)
                            typical_oracle_kappa = SeparableOracle(data, master, ParetoOracle, data.num_scenarios; model = update_sub_model!, sub_oracle_param = pareto_param, optimizer = optimizer)
                            typical_oracle_nu = SeparableOracle(data, master, ParetoOracle, data.num_scenarios; model = update_sub_model!, sub_oracle_param = pareto_param, optimizer = optimizer)
                            typical_oracles = [typical_oracle_kappa; typical_oracle_nu]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); param = oracle_param)

                            preprocessing = NoPreprocessing()
                            lazy_callback = LazyCallback(lazy_oracle)
                            user_callback = UserCallback(disjunctive_oracle; param=user_cb_param)

                            env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                            log = solve!(env)
                            @test env.termination_status == Optimal()
                            @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                            if benders_param.verbose
                                @info "Disjunctive cuts added: $(length(env.user_callback.oracle.disjunctive_cuts))"
                                env.user_callback.oracle.param.add_benders_cuts_to_master != 0 && @info "Byproduct Benders cuts added: $(log.n_user_cuts[1] - length(env.user_callback.oracle.disjunctive_cuts))"
                            end
                        end

                        @testset "Seq" begin
                            @info "solving SNIP instance$instance snipno $snipno budget $budget - disjunctive oracle/pareto/seq - strgthnd $strengthened; benders2master $add_benders_cuts_to_master reuse $reuse_dcglp lift $lift p $p dcut_append $disjunctive_cut_append_rule"
                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            pareto_param = ParetoOracleParam(ones(length(data.D)))
                            lazy_oracle = SeparableOracle(data, master, ParetoOracle, data.num_scenarios; model = update_sub_model!, sub_oracle_param = pareto_param, optimizer = optimizer)
                            typical_oracle_kappa = SeparableOracle(data, master, ParetoOracle, data.num_scenarios; model = update_sub_model!, sub_oracle_param = pareto_param, optimizer = optimizer)
                            typical_oracle_nu = SeparableOracle(data, master, ParetoOracle, data.num_scenarios; model = update_sub_model!, sub_oracle_param = pareto_param, optimizer = optimizer)
                            typical_oracles = [typical_oracle_kappa; typical_oracle_nu]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); param = oracle_param)

                            preprocessing = LPRelaxationPreprocessing(lazy_oracle; seq_env_type = BendersSeq, param = BendersSeqParam(;time_limit=200.0, gap_tolerance=1e-9, verbose=false))
                            lazy_callback = LazyCallback(lazy_oracle)
                            user_callback = UserCallback(disjunctive_oracle; param=user_cb_param)

                            env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                            log = solve!(env)
                            @test env.termination_status == Optimal()
                            @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                            if benders_param.verbose
                                @info "Disjunctive cuts added: $(length(env.user_callback.oracle.disjunctive_cuts))"
                                env.user_callback.oracle.param.add_benders_cuts_to_master != 0 && @info "Byproduct Benders cuts added: $(log.n_user_cuts[1] - length(env.user_callback.oracle.disjunctive_cuts))"
                            end
                        end

                        @testset "SeqInOut" begin
                            @info "solving SNIP instance$instance snipno $snipno budget $budget - disjunctive oracle/pareto/seqinout - strgthnd $strengthened; benders2master $add_benders_cuts_to_master reuse $reuse_dcglp lift $lift p $p dcut_append $disjunctive_cut_append_rule"
                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            pareto_param = ParetoOracleParam(ones(length(data.D)))
                            lazy_oracle = SeparableOracle(data, master, ParetoOracle, data.num_scenarios; model = update_sub_model!, sub_oracle_param = pareto_param, optimizer = optimizer)
                            typical_oracle_kappa = SeparableOracle(data, master, ParetoOracle, data.num_scenarios; model = update_sub_model!, sub_oracle_param = pareto_param, optimizer = optimizer)
                            typical_oracle_nu = SeparableOracle(data, master, ParetoOracle, data.num_scenarios; model = update_sub_model!, sub_oracle_param = pareto_param, optimizer = optimizer)
                            typical_oracles = [typical_oracle_kappa; typical_oracle_nu]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); param = oracle_param)

                            preprocessing = LPRelaxationPreprocessing(lazy_oracle; seq_env_type = BendersSeqInOut, param = BendersSeqInOutParam(time_limit = 300.0, gap_tolerance = 1e-9, stabilizing_x = ones(length(data.D)), α = 0.9, λ = 0.1, verbose = false))
                            lazy_callback = LazyCallback(lazy_oracle)
                            user_callback = UserCallback(disjunctive_oracle; param=user_cb_param)

                            env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                            log = solve!(env)
                            @test env.termination_status == Optimal()
                            @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                            if benders_param.verbose
                                @info "Disjunctive cuts added: $(length(env.user_callback.oracle.disjunctive_cuts))"
                                env.user_callback.oracle.param.add_benders_cuts_to_master != 0 && @info "Byproduct Benders cuts added: $(log.n_user_cuts[1] - length(env.user_callback.oracle.disjunctive_cuts))"
                            end
                        end
                    end
                end
            end

            @testset "Unified oracle" begin
                for strengthened in [true], add_benders_cuts_to_master in [true], reuse_dcglp in [true], p in [Inf], lift in [true], disjunctive_cut_append_rule in [AllDisjunctiveCuts()]
                    @testset "strgthnd $strengthened; benders2master $add_benders_cuts_to_master; reuse $reuse_dcglp; p $p; lift $lift; dcut_append $disjunctive_cut_append_rule" begin

                        oracle_param = SplitOracleParam(; normalization = LpDistanceNormalization(p), dcglp_param = dcglp_param,
                                                            split_index_selection_rule = LargestFractional(),
                                                            disjunctive_cut_append_rule = disjunctive_cut_append_rule,
                                                            strengthened = strengthened,
                                                            add_benders_cuts_to_master = add_benders_cuts_to_master,
                                                            fraction_of_benders_cuts_to_master = 1.0,
                                                            reuse_dcglp = reuse_dcglp,
                                                            lift = lift)

                        @testset "NoSeq" begin
                            @info "solving SNIP instance$instance snipno $snipno budget $budget - disjunctive oracle/unified/no seq - strgthnd $strengthened; benders2master $add_benders_cuts_to_master reuse $reuse_dcglp lift $lift p $p dcut_append $disjunctive_cut_append_rule"
                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            unified_param = UnifiedOracleParam()
                            lazy_oracle = SeparableOracle(data, master, UnifiedOracle, data.num_scenarios; model = update_sub_model!, sub_oracle_param = unified_param, optimizer = optimizer)
                            typical_oracle_kappa = SeparableOracle(data, master, UnifiedOracle, data.num_scenarios; model = update_sub_model!, sub_oracle_param = unified_param, optimizer = optimizer)
                            typical_oracle_nu = SeparableOracle(data, master, UnifiedOracle, data.num_scenarios; model = update_sub_model!, sub_oracle_param = unified_param, optimizer = optimizer)
                            typical_oracles = [typical_oracle_kappa; typical_oracle_nu]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); param = oracle_param)

                            preprocessing = NoPreprocessing()
                            lazy_callback = LazyCallback(lazy_oracle)
                            user_callback = UserCallback(disjunctive_oracle; param=user_cb_param)

                            env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                            log = solve!(env)
                            @test env.termination_status == Optimal()
                            @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                            if benders_param.verbose
                                @info "Disjunctive cuts added: $(length(env.user_callback.oracle.disjunctive_cuts))"
                                env.user_callback.oracle.param.add_benders_cuts_to_master != 0 && @info "Byproduct Benders cuts added: $(log.n_user_cuts[1] - length(env.user_callback.oracle.disjunctive_cuts))"
                            end
                        end

                        @testset "Seq" begin
                            @info "solving SNIP instance$instance snipno $snipno budget $budget - disjunctive oracle/unified/seq - strgthnd $strengthened; benders2master $add_benders_cuts_to_master reuse $reuse_dcglp lift $lift p $p dcut_append $disjunctive_cut_append_rule"
                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            unified_param = UnifiedOracleParam()
                            lazy_oracle = SeparableOracle(data, master, UnifiedOracle, data.num_scenarios; model = update_sub_model!, sub_oracle_param = unified_param, optimizer = optimizer)
                            typical_oracle_kappa = SeparableOracle(data, master, UnifiedOracle, data.num_scenarios; model = update_sub_model!, sub_oracle_param = unified_param, optimizer = optimizer)
                            typical_oracle_nu = SeparableOracle(data, master, UnifiedOracle, data.num_scenarios; model = update_sub_model!, sub_oracle_param = unified_param, optimizer = optimizer)
                            typical_oracles = [typical_oracle_kappa; typical_oracle_nu]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); param = oracle_param)

                            preprocessing = LPRelaxationPreprocessing(lazy_oracle; seq_env_type = BendersSeq, param = BendersSeqParam(;time_limit=200.0, gap_tolerance=1e-9, verbose=false))
                            lazy_callback = LazyCallback(lazy_oracle)
                            user_callback = UserCallback(disjunctive_oracle; param=user_cb_param)

                            env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                            log = solve!(env)
                            @test env.termination_status == Optimal()
                            @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                            if benders_param.verbose
                                @info "Disjunctive cuts added: $(length(env.user_callback.oracle.disjunctive_cuts))"
                                env.user_callback.oracle.param.add_benders_cuts_to_master != 0 && @info "Byproduct Benders cuts added: $(log.n_user_cuts[1] - length(env.user_callback.oracle.disjunctive_cuts))"
                            end
                        end

                        @testset "SeqInOut" begin
                            @info "solving SNIP instance$instance snipno $snipno budget $budget - disjunctive oracle/unified/seqinout - strgthnd $strengthened; benders2master $add_benders_cuts_to_master reuse $reuse_dcglp lift $lift p $p dcut_append $disjunctive_cut_append_rule"
                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            unified_param = UnifiedOracleParam()
                            lazy_oracle = SeparableOracle(data, master, UnifiedOracle, data.num_scenarios; model = update_sub_model!, sub_oracle_param = unified_param, optimizer = optimizer)
                            typical_oracle_kappa = SeparableOracle(data, master, UnifiedOracle, data.num_scenarios; model = update_sub_model!, sub_oracle_param = unified_param, optimizer = optimizer)
                            typical_oracle_nu = SeparableOracle(data, master, UnifiedOracle, data.num_scenarios; model = update_sub_model!, sub_oracle_param = unified_param, optimizer = optimizer)
                            typical_oracles = [typical_oracle_kappa; typical_oracle_nu]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); param = oracle_param)

                            preprocessing = LPRelaxationPreprocessing(lazy_oracle; seq_env_type = BendersSeqInOut, param = BendersSeqInOutParam(time_limit = 300.0, gap_tolerance = 1e-9, stabilizing_x = ones(length(data.D)), α = 0.9, λ = 0.1, verbose = false))
                            lazy_callback = LazyCallback(lazy_oracle)
                            user_callback = UserCallback(disjunctive_oracle; param=user_cb_param)

                            env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                            log = solve!(env)
                            @test env.termination_status == Optimal()
                            @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                            if benders_param.verbose
                                @info "Disjunctive cuts added: $(length(env.user_callback.oracle.disjunctive_cuts))"
                                env.user_callback.oracle.param.add_benders_cuts_to_master != 0 && @info "Byproduct Benders cuts added: $(log.n_user_cuts[1] - length(env.user_callback.oracle.disjunctive_cuts))"
                            end
                        end
                    end
                end
            end

        end
    end
end
