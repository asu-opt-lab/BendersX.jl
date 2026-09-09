using BendersX
using CSV
using DataFrames
using Test
using JuMP
using CPLEX
include(normpath(joinpath(@__DIR__, "..", "solver_defaults.jl")))

@testset verbose = true "CFLP Disjunctive Callback Benders Tests" begin
    reference_path = normpath(joinpath(@__DIR__, "..", "reference_objectives", "cflp.csv"))
    reference_df = DataFrame(CSV.File(reference_path))
    @assert nrow(reference_df) == length(unique(reference_df.instance_name)) "Duplicate CFLP reference objectives found in $(reference_path)"
    reference_objectives = Dict(String(row.instance_name) => Float64(row.objective_value) for row in eachrow(reference_df))
    instances = setdiff(1:71, [67])

    # GBC-enabled subproblem model update (y[i,j] <= x[i] via GBC)
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
            # Load problem data
            data = read_cflp_benchmark_data("p$i")

            # Algorithm parameters
            benders_param = BendersBnBParam(;
                                            time_limit = 200.0,
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

            user_cb_param = UserCallbackParam(frequency=1)

            instance_name = "p$i"
            @assert haskey(reference_objectives, instance_name) "Missing CFLP reference objective for $(instance_name) in $(reference_path)"
            mip_opt_val = reference_objectives[instance_name]

            @testset "Classic oracle" begin
                # for strengthened in [true; false], add_benders_cuts_to_master in [true; false; 2], reuse_dcglp in [true; false], p in [1.0; Inf], lift in [true; false], disjunctive_cut_append_rule in [NoDisjunctiveCuts(); AllDisjunctiveCuts(); DisjunctiveCutsSmallerIndices()]
                for strengthened in [true], add_benders_cuts_to_master in [true], reuse_dcglp in [true], p in [1.0], lift in [true], disjunctive_cut_append_rule in [AllDisjunctiveCuts()]
                    @testset "strgthnd $strengthened; benders2master $add_benders_cuts_to_master; reuse $reuse_dcglp; lift $lift; p $p; lift $lift; dcut_append $disjunctive_cut_append_rule" begin
                        oracle_param = SplitOracleParam(; dcglp_param = dcglp_param,
                                                                split_index_selection_rule = RandomFractional(),
                                                                disjunctive_cut_append_rule = disjunctive_cut_append_rule,
                                                                strengthened = strengthened,
                                                                add_benders_cuts_to_master = add_benders_cuts_to_master,
                                                                fraction_of_benders_cuts_to_master = 0.5,
                                                                reuse_dcglp = reuse_dcglp,
                                                                lift = lift)

                        @testset "NoSeq" begin
                            @info "solving CFLP p$i - disjunctive oracle/classical/no seq - strgthnd $strengthened; benders2master $add_benders_cuts_to_master reuse $reuse_dcglp lift $lift p $p dcut_append $disjunctive_cut_append_rule"
                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            lazy_oracle = ClassicalOracle(data, master; model = update_sub_model!, optimizer = optimizer)
                            typical_oracles = [ClassicalOracle(data, master; model = update_sub_model!, optimizer = optimizer), ClassicalOracle(data, master; model = update_sub_model!, optimizer = optimizer)]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); normalization = LpDistanceNormalization(p), param = oracle_param)

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
                            @info "solving CFLP p$i - disjunctive oracle/classical/seq - strgthnd $strengthened; benders2master $add_benders_cuts_to_master reuse $reuse_dcglp lift $lift p $p dcut_append $disjunctive_cut_append_rule"
                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            lazy_oracle = ClassicalOracle(data, master; model = update_sub_model!, optimizer = optimizer)
                            typical_oracles = [ClassicalOracle(data, master; model = update_sub_model!, optimizer = optimizer), ClassicalOracle(data, master; model = update_sub_model!, optimizer = optimizer)]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); normalization = LpDistanceNormalization(p), param = oracle_param)

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
                            @info "solving CFLP p$i - disjunctive oracle/classical/seqinout - strgthnd $strengthened; benders2master $add_benders_cuts_to_master reuse $reuse_dcglp lift $lift p $p dcut_append $disjunctive_cut_append_rule"
                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            lazy_oracle = ClassicalOracle(data, master; model = update_sub_model!, optimizer = optimizer)
                            typical_oracles = [ClassicalOracle(data, master; model = update_sub_model!, optimizer = optimizer), ClassicalOracle(data, master; model = update_sub_model!, optimizer = optimizer)]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); normalization = LpDistanceNormalization(p), param = oracle_param)

                            preprocessing = LPRelaxationPreprocessing(lazy_oracle; seq_env_type = BendersSeqInOut, param = BendersSeqInOutParam(time_limit = 300.0, gap_tolerance = 1e-9, stabilizing_x = ones(data.n_facilities), α = 0.9, λ = 0.1, verbose = false))
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

            @testset "Knapsack oracle" begin
                # for strengthened in [true; false], add_benders_cuts_to_master in [true; false; 2], reuse_dcglp in [true; false], p in [1.0; Inf], lift in [true; false], disjunctive_cut_append_rule in [NoDisjunctiveCuts(); AllDisjunctiveCuts(); DisjunctiveCutsSmallerIndices()]
                for strengthened in [true], add_benders_cuts_to_master in [true], reuse_dcglp in [true], p in [1.0], lift in [true], disjunctive_cut_append_rule in [AllDisjunctiveCuts()]
                    @testset "strgthnd $strengthened; benders2master $add_benders_cuts_to_master; reuse $reuse_dcglp; lift $lift; p $p; dcut_append $disjunctive_cut_append_rule" begin

                        oracle_param = SplitOracleParam(; dcglp_param = dcglp_param,
                                                                split_index_selection_rule = RandomFractional(),
                                                                disjunctive_cut_append_rule = disjunctive_cut_append_rule,
                                                                strengthened = strengthened,
                                                                add_benders_cuts_to_master = add_benders_cuts_to_master,
                                                                fraction_of_benders_cuts_to_master = 0.5,
                                                                reuse_dcglp = reuse_dcglp,
                                                                lift = lift)

                        @testset "NoSeq" begin
                            @info "solving CFLP p$i - disjunctive oracle/knapsack oracle/no seq - strgthnd $strengthened; benders2master $add_benders_cuts_to_master reuse $reuse_dcglp lift $lift p $p dcut_append $disjunctive_cut_append_rule"
                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            lazy_oracle = CFLKnapsackOracle(data, master; model = update_sub_model!, optimizer = optimizer)
                            typical_oracles = [CFLKnapsackOracle(data, master; model = update_sub_model!, optimizer = optimizer), CFLKnapsackOracle(data, master; model = update_sub_model!, optimizer = optimizer)]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); normalization = LpDistanceNormalization(p), param = oracle_param)

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
                            @info "solving CFLP p$i - disjunctive oracle/knapsack oracle/seq - strgthnd $strengthened; benders2master $add_benders_cuts_to_master reuse $reuse_dcglp lift $lift p $p dcut_append $disjunctive_cut_append_rule"
                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            lazy_oracle = CFLKnapsackOracle(data, master; model = update_sub_model!, optimizer = optimizer)
                            typical_oracles = [CFLKnapsackOracle(data, master; model = update_sub_model!, optimizer = optimizer), CFLKnapsackOracle(data, master; model = update_sub_model!, optimizer = optimizer)]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); normalization = LpDistanceNormalization(p), param = oracle_param)

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
                            @info "solving CFLP p$i - disjunctive oracle/knapsack oracle/seqinout - strgthnd $strengthened; benders2master $add_benders_cuts_to_master reuse $reuse_dcglp lift $lift p $p dcut_append $disjunctive_cut_append_rule"
                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            lazy_oracle = CFLKnapsackOracle(data, master; model = update_sub_model!, optimizer = optimizer)
                            typical_oracles = [CFLKnapsackOracle(data, master; model = update_sub_model!, optimizer = optimizer), CFLKnapsackOracle(data, master; model = update_sub_model!, optimizer = optimizer)]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); normalization = LpDistanceNormalization(p), param = oracle_param)

                            preprocessing = LPRelaxationPreprocessing(lazy_oracle; seq_env_type = BendersSeqInOut, param = BendersSeqInOutParam(time_limit = 300.0, gap_tolerance = 1e-9, stabilizing_x = ones(data.n_facilities), α = 0.9, λ = 0.1, verbose = false))
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
                for strengthened in [true], add_benders_cuts_to_master in [true], reuse_dcglp in [true], p in [1.0], lift in [true], disjunctive_cut_append_rule in [AllDisjunctiveCuts()]
                    @testset "strgthnd $strengthened; benders2master $add_benders_cuts_to_master; reuse $reuse_dcglp; p $p; lift $lift; dcut_append $disjunctive_cut_append_rule" begin
                        oracle_param = SplitOracleParam(; dcglp_param = dcglp_param,
                                                                split_index_selection_rule = RandomFractional(),
                                                                disjunctive_cut_append_rule = disjunctive_cut_append_rule,
                                                                strengthened = strengthened,
                                                                add_benders_cuts_to_master = add_benders_cuts_to_master,
                                                                fraction_of_benders_cuts_to_master = 0.5,
                                                                reuse_dcglp = reuse_dcglp,
                                                                lift = lift)

                        @testset "NoSeq" begin
                            @info "solving CFLP p$i - disjunctive oracle/unified/no seq"
                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            lazy_oracle = UnifiedOracle(data, master; model = update_sub_model!, optimizer = optimizer)
                            typical_oracles = [UnifiedOracle(data, master; model = update_sub_model!, optimizer = optimizer), UnifiedOracle(data, master; model = update_sub_model!, optimizer = optimizer)]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); normalization = LpDistanceNormalization(p), param = oracle_param)

                            preprocessing = NoPreprocessing()
                            lazy_callback = LazyCallback(lazy_oracle)
                            user_callback = UserCallback(disjunctive_oracle; param=user_cb_param)

                            env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                            log = solve!(env)
                            @test env.termination_status == Optimal()
                            @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                        end

                        @testset "Seq" begin
                            @info "solving CFLP p$i - disjunctive oracle/unified/seq"
                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            lazy_oracle = UnifiedOracle(data, master; model = update_sub_model!, optimizer = optimizer)
                            typical_oracles = [UnifiedOracle(data, master; model = update_sub_model!, optimizer = optimizer), UnifiedOracle(data, master; model = update_sub_model!, optimizer = optimizer)]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); normalization = LpDistanceNormalization(p), param = oracle_param)

                            preprocessing = LPRelaxationPreprocessing(lazy_oracle; seq_env_type = BendersSeq, param = BendersSeqParam(;time_limit=200.0, gap_tolerance=1e-9, verbose=false))
                            lazy_callback = LazyCallback(lazy_oracle)
                            user_callback = UserCallback(disjunctive_oracle; param=user_cb_param)

                            env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                            log = solve!(env)
                            @test env.termination_status == Optimal()
                            @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                        end

                        @testset "SeqInOut" begin
                            @info "solving CFLP p$i - disjunctive oracle/unified/seqinout"
                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            lazy_oracle = UnifiedOracle(data, master; model = update_sub_model!, optimizer = optimizer)
                            typical_oracles = [UnifiedOracle(data, master; model = update_sub_model!, optimizer = optimizer), UnifiedOracle(data, master; model = update_sub_model!, optimizer = optimizer)]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); normalization = LpDistanceNormalization(p), param = oracle_param)

                            preprocessing = LPRelaxationPreprocessing(lazy_oracle; seq_env_type = BendersSeqInOut, param = BendersSeqInOutParam(time_limit = 300.0, gap_tolerance = 1e-9, stabilizing_x = ones(data.n_facilities), α = 0.9, λ = 0.1, verbose = false))
                            lazy_callback = LazyCallback(lazy_oracle)
                            user_callback = UserCallback(disjunctive_oracle; param=user_cb_param)

                            env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                            log = solve!(env)
                            @test env.termination_status == Optimal()
                            @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                        end
                    end
                end
            end

            @testset "Pareto oracle" begin
                for strengthened in [true], add_benders_cuts_to_master in [true], reuse_dcglp in [true], p in [1.0], lift in [true], disjunctive_cut_append_rule in [AllDisjunctiveCuts()]
                    @testset "strgthnd $strengthened; benders2master $add_benders_cuts_to_master; reuse $reuse_dcglp; p $p; lift $lift; dcut_append $disjunctive_cut_append_rule" begin
                        oracle_param = SplitOracleParam(; dcglp_param = dcglp_param,
                                                                split_index_selection_rule = RandomFractional(),
                                                                disjunctive_cut_append_rule = disjunctive_cut_append_rule,
                                                                strengthened = strengthened,
                                                                add_benders_cuts_to_master = add_benders_cuts_to_master,
                                                                fraction_of_benders_cuts_to_master = 0.5,
                                                                reuse_dcglp = reuse_dcglp,
                                                                lift = lift)

                        @testset "NoSeq" begin
                            @info "solving CFLP p$i - disjunctive oracle/pareto/no seq"
                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            pareto_param = ParetoOracleParam(ones(data.n_facilities))
                            lazy_oracle = ParetoOracle(data, master, pareto_param; model = update_sub_model!, optimizer = optimizer)
                            typical_oracles = [ParetoOracle(data, master, pareto_param; model = update_sub_model!, optimizer = optimizer), ParetoOracle(data, master, pareto_param; model = update_sub_model!, optimizer = optimizer)]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); normalization = LpDistanceNormalization(p), param = oracle_param)

                            preprocessing = NoPreprocessing()
                            lazy_callback = LazyCallback(lazy_oracle)
                            user_callback = UserCallback(disjunctive_oracle; param=user_cb_param)

                            env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                            log = solve!(env)
                            @test env.termination_status == Optimal()
                            @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                        end

                        @testset "Seq" begin
                            @info "solving CFLP p$i - disjunctive oracle/pareto/seq"
                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            pareto_param = ParetoOracleParam(ones(data.n_facilities))
                            lazy_oracle = ParetoOracle(data, master, pareto_param; model = update_sub_model!, optimizer = optimizer)
                            typical_oracles = [ParetoOracle(data, master, pareto_param; model = update_sub_model!, optimizer = optimizer), ParetoOracle(data, master, pareto_param; model = update_sub_model!, optimizer = optimizer)]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); normalization = LpDistanceNormalization(p), param = oracle_param)

                            preprocessing = LPRelaxationPreprocessing(lazy_oracle; seq_env_type = BendersSeq, param = BendersSeqParam(;time_limit=200.0, gap_tolerance=1e-9, verbose=false))
                            lazy_callback = LazyCallback(lazy_oracle)
                            user_callback = UserCallback(disjunctive_oracle; param=user_cb_param)

                            env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                            log = solve!(env)
                            @test env.termination_status == Optimal()
                            @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                        end

                        @testset "SeqInOut" begin
                            @info "solving CFLP p$i - disjunctive oracle/pareto/seqinout"
                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            pareto_param = ParetoOracleParam(ones(data.n_facilities))
                            lazy_oracle = ParetoOracle(data, master, pareto_param; model = update_sub_model!, optimizer = optimizer)
                            typical_oracles = [ParetoOracle(data, master, pareto_param; model = update_sub_model!, optimizer = optimizer), ParetoOracle(data, master, pareto_param; model = update_sub_model!, optimizer = optimizer)]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); normalization = LpDistanceNormalization(p), param = oracle_param)

                            preprocessing = LPRelaxationPreprocessing(lazy_oracle; seq_env_type = BendersSeqInOut, param = BendersSeqInOutParam(time_limit = 300.0, gap_tolerance = 1e-9, stabilizing_x = ones(data.n_facilities), α = 0.9, λ = 0.1, verbose = false))
                            lazy_callback = LazyCallback(lazy_oracle)
                            user_callback = UserCallback(disjunctive_oracle; param=user_cb_param)

                            env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                            log = solve!(env)
                            @test env.termination_status == Optimal()
                            @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                        end
                    end
                end
            end

            @testset "Classic oracle with GBC" begin
                for strengthened in [true], add_benders_cuts_to_master in [true], reuse_dcglp in [true], p in [1.0], lift in [true], disjunctive_cut_append_rule in [AllDisjunctiveCuts()]
                    @testset "strgthnd $strengthened; benders2master $add_benders_cuts_to_master; reuse $reuse_dcglp; p $p; lift $lift; dcut_append $disjunctive_cut_append_rule" begin
                        oracle_param = SplitOracleParam(; dcglp_param = dcglp_param,
                                                                split_index_selection_rule = RandomFractional(),
                                                                disjunctive_cut_append_rule = disjunctive_cut_append_rule,
                                                                strengthened = strengthened,
                                                                add_benders_cuts_to_master = add_benders_cuts_to_master,
                                                                fraction_of_benders_cuts_to_master = 0.5,
                                                                reuse_dcglp = reuse_dcglp,
                                                                lift = lift)

                        @testset "NoSeq" begin
                            @info "solving CFLP p$i - disjunctive oracle/classical with GBC/no seq"
                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            lazy_oracle = ClassicalOracle(data, master; model = update_sub_gbc_model!, optimizer = optimizer)
                            typical_oracles = [ClassicalOracle(data, master; model = update_sub_gbc_model!, optimizer = optimizer), ClassicalOracle(data, master; model = update_sub_gbc_model!, optimizer = optimizer)]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); normalization = LpDistanceNormalization(p), param = oracle_param)

                            preprocessing = NoPreprocessing()
                            lazy_callback = LazyCallback(lazy_oracle)
                            user_callback = UserCallback(disjunctive_oracle; param=user_cb_param)

                            env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                            log = solve!(env)
                            @test env.termination_status == Optimal()
                            @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                        end

                        @testset "Seq" begin
                            @info "solving CFLP p$i - disjunctive oracle/classical with GBC/seq"
                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            lazy_oracle = ClassicalOracle(data, master; model = update_sub_gbc_model!, optimizer = optimizer)
                            typical_oracles = [ClassicalOracle(data, master; model = update_sub_gbc_model!, optimizer = optimizer), ClassicalOracle(data, master; model = update_sub_gbc_model!, optimizer = optimizer)]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); normalization = LpDistanceNormalization(p), param = oracle_param)

                            preprocessing = LPRelaxationPreprocessing(lazy_oracle; seq_env_type = BendersSeq, param = BendersSeqParam(;time_limit=200.0, gap_tolerance=1e-9, verbose=false))
                            lazy_callback = LazyCallback(lazy_oracle)
                            user_callback = UserCallback(disjunctive_oracle; param=user_cb_param)

                            env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                            log = solve!(env)
                            @test env.termination_status == Optimal()
                            @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                        end

                        @testset "SeqInOut" begin
                            @info "solving CFLP p$i - disjunctive oracle/classical with GBC/seqinout"
                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            lazy_oracle = ClassicalOracle(data, master; model = update_sub_gbc_model!, optimizer = optimizer)
                            typical_oracles = [ClassicalOracle(data, master; model = update_sub_gbc_model!, optimizer = optimizer), ClassicalOracle(data, master; model = update_sub_gbc_model!, optimizer = optimizer)]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); normalization = LpDistanceNormalization(p), param = oracle_param)

                            preprocessing = LPRelaxationPreprocessing(lazy_oracle; seq_env_type = BendersSeqInOut, param = BendersSeqInOutParam(time_limit = 300.0, gap_tolerance = 1e-9, stabilizing_x = ones(data.n_facilities), α = 0.9, λ = 0.1, verbose = false))
                            lazy_callback = LazyCallback(lazy_oracle)
                            user_callback = UserCallback(disjunctive_oracle; param=user_cb_param)

                            env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                            log = solve!(env)
                            @test env.termination_status == Optimal()
                            @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                        end
                    end
                end
            end

            @testset "Knapsack oracle with GBC" begin
                for strengthened in [true], add_benders_cuts_to_master in [true], reuse_dcglp in [true], p in [1.0], lift in [true], disjunctive_cut_append_rule in [AllDisjunctiveCuts()]
                    @testset "strgthnd $strengthened; benders2master $add_benders_cuts_to_master; reuse $reuse_dcglp; p $p; lift $lift; dcut_append $disjunctive_cut_append_rule" begin
                        oracle_param = SplitOracleParam(; dcglp_param = dcglp_param,
                                                                split_index_selection_rule = RandomFractional(),
                                                                disjunctive_cut_append_rule = disjunctive_cut_append_rule,
                                                                strengthened = strengthened,
                                                                add_benders_cuts_to_master = add_benders_cuts_to_master,
                                                                fraction_of_benders_cuts_to_master = 0.5,
                                                                reuse_dcglp = reuse_dcglp,
                                                                lift = lift)

                        @testset "NoSeq" begin
                            @info "solving CFLP p$i - disjunctive oracle/knapsack oracle with GBC/no seq"
                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            lazy_oracle = CFLKnapsackOracle(data, master; model = update_sub_gbc_model!, optimizer = optimizer)
                            typical_oracles = [CFLKnapsackOracle(data, master; model = update_sub_gbc_model!, optimizer = optimizer), CFLKnapsackOracle(data, master; model = update_sub_gbc_model!, optimizer = optimizer)]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); normalization = LpDistanceNormalization(p), param = oracle_param)

                            preprocessing = NoPreprocessing()
                            lazy_callback = LazyCallback(lazy_oracle)
                            user_callback = UserCallback(disjunctive_oracle; param=user_cb_param)

                            env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                            log = solve!(env)
                            @test env.termination_status == Optimal()
                            @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                        end

                        @testset "Seq" begin
                            @info "solving CFLP p$i - disjunctive oracle/knapsack oracle with GBC/seq"
                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            lazy_oracle = CFLKnapsackOracle(data, master; model = update_sub_gbc_model!, optimizer = optimizer)
                            typical_oracles = [CFLKnapsackOracle(data, master; model = update_sub_gbc_model!, optimizer = optimizer), CFLKnapsackOracle(data, master; model = update_sub_gbc_model!, optimizer = optimizer)]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); normalization = LpDistanceNormalization(p), param = oracle_param)

                            preprocessing = LPRelaxationPreprocessing(lazy_oracle; seq_env_type = BendersSeq, param = BendersSeqParam(;time_limit=200.0, gap_tolerance=1e-9, verbose=false))
                            lazy_callback = LazyCallback(lazy_oracle)
                            user_callback = UserCallback(disjunctive_oracle; param=user_cb_param)

                            env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                            log = solve!(env)
                            @test env.termination_status == Optimal()
                            @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                        end

                        @testset "SeqInOut" begin
                            @info "solving CFLP p$i - disjunctive oracle/knapsack oracle with GBC/seqinout"
                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            lazy_oracle = CFLKnapsackOracle(data, master; model = update_sub_gbc_model!, optimizer = optimizer)
                            typical_oracles = [CFLKnapsackOracle(data, master; model = update_sub_gbc_model!, optimizer = optimizer), CFLKnapsackOracle(data, master; model = update_sub_gbc_model!, optimizer = optimizer)]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); normalization = LpDistanceNormalization(p), param = oracle_param)

                            preprocessing = LPRelaxationPreprocessing(lazy_oracle; seq_env_type = BendersSeqInOut, param = BendersSeqInOutParam(time_limit = 300.0, gap_tolerance = 1e-9, stabilizing_x = ones(data.n_facilities), α = 0.9, λ = 0.1, verbose = false))
                            lazy_callback = LazyCallback(lazy_oracle)
                            user_callback = UserCallback(disjunctive_oracle; param=user_cb_param)

                            env = BendersBnB(master; preprocessing = preprocessing, lazy_callback = lazy_callback, user_callback = user_callback, param = benders_param)
                            log = solve!(env)
                            @test env.termination_status == Optimal()
                            @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                        end
                    end
                end
            end
        end
    end
end
