using BendersX
using CSV
using DataFrames
using Test
using JuMP
using CPLEX
include(normpath(joinpath(@__DIR__, "..", "solver_defaults.jl")))

@testset verbose = true "UFLP Sequential Benders Tests -- MIP master" begin
    reference_path = normpath(joinpath(@__DIR__, "..", "reference_objectives", "uflp.csv"))
    reference_df = DataFrame(CSV.File(reference_path))
    @assert nrow(reference_df) == length(unique(reference_df.instance_name)) "Duplicate UFLP reference objectives found in $(reference_path)"
    reference_objectives = Dict(String(row.instance_name) => Float64(row.objective_value) for row in eachrow(reference_df))
    instances = setdiff(1:71, [67])

    # GBC-enabled subproblem model update (y[i,j] <= x[i] via GBC)
    function update_sub_gbc_model!(model::Model, data::UFLPData, scen_idx::Int; x)
        optimizer = optimizer_with_attributes(CPLEX.Optimizer, "CPXPARAM_Threads" => 7, "CPX_PARAM_EPRHS" => 1e-9, "CPX_PARAM_EPOPT" => 1e-9, "CPX_PARAM_NUMERICALEMPHASIS" => 1, MOI.Silent() => true)
        set_optimizer(model, optimizer)
        I, J = data.n_facilities, data.n_customers
        @variable(model, y[1:I, 1:J] >= 0)
        cost_demands = data.costs .* data.demands'
        @objective(model, Min, sum(cost_demands .* y))
        @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
        gbc_lhs = vec(y)
        gbc_rhs = [x[i] for j in 1:J for i in 1:I]
        gbc_sense = fill(UpperBound, I*J)
        return gbc_lhs, gbc_rhs, gbc_sense
    end

    for i in instances
        @testset "Instance: p$i" begin
            # Load problem data
            data = read_uflp_benchmark_data("p$(i)")

            # Algorithm parameters
            benders_param = BendersSeqParam(;
                            time_limit = 800.0,
                            gap_tolerance = 1e-6,
                            verbose = false
                            )

            dcglp_optimizer = optimizer_with_attributes(CPLEX.Optimizer, "CPXPARAM_Threads" => 7, "CPX_PARAM_EPRHS" => 1e-9, "CPX_PARAM_NUMERICALEMPHASIS" => 1, "CPX_PARAM_EPOPT" => 1e-9, MOI.Silent() => true)
            dcglp_param = DcglpParam(; optimizer = dcglp_optimizer,
                                    time_limit = 1000.0,
                                    gap_tolerance = 1e-3,
                                    halt_limit = 3,
                                    iter_limit = 250,
                                    verbose = false
                                    )

            instance_name = "p$i"
            @assert haskey(reference_objectives, instance_name) "Missing UFLP reference objective for $(instance_name) in $(reference_path)"
            mip_opt_val = reference_objectives[instance_name]

            @testset "DCGLP normalization oracles" begin
                normalization_specs = [
                    (
                        "SplitOracle/ReversePolarNormalization",
                        ReversePolarNormalization(),
                        SplitOracleParam(; dcglp_param = dcglp_param,
                            split_index_selection_rule = LargestFractional(),
                            disjunctive_cut_append_rule = AllDisjunctiveCuts(),
                            strengthened = true,
                            add_benders_cuts_to_master = false,
                            reuse_dcglp = false,
                            lift = false,
                            zero_tol = 1e-9,
                        ),
                    ),
                ]

                for (normalization_name, normalization, oracle_param) in normalization_specs
                    @info "solving UFLP p$i - $normalization_name/classical - benders2master false reuse false lift false"
                    @testset "$normalization_name with ClassicalOracle" begin
                        master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                        typical_oracles = [
                            ClassicalOracle(data, master; model = update_sub_model!, optimizer = optimizer),
                            ClassicalOracle(data, master; model = update_sub_model!, optimizer = optimizer),
                        ]
                        disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); normalization = normalization, param = oracle_param)

                        env = BendersSeq(master, disjunctive_oracle; param = benders_param)
                        solve!(env)
                        @test env.termination_status == Optimal()
                        @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                    end
                end
            end

            @testset "Classical oracle" begin
                @testset "Seq" begin
                    # for strengthened in [true; false], add_benders_cuts_to_master in [true; false; 2], reuse_dcglp in [true; false], p in [1.0; Inf], lift in [true; false], disjunctive_cut_append_rule in [NoDisjunctiveCuts(); AllDisjunctiveCuts(); DisjunctiveCutsSmallerIndices()]
                    for strengthened in [true], add_benders_cuts_to_master in [2], reuse_dcglp in [false], p in [Inf], lift in [false], disjunctive_cut_append_rule in [AllDisjunctiveCuts()]
                        @info "solving UFLP p$i - disjunctive oracle/classical - strgthnd $strengthened; benders2master $add_benders_cuts_to_master reuse $reuse_dcglp p $p lift $lift dcut_append $disjunctive_cut_append_rule"
                        @testset "strgthnd $strengthened; benders2master $add_benders_cuts_to_master; reuse $reuse_dcglp; p $p; lift $lift; dcut_append $disjunctive_cut_append_rule" begin

                            oracle_param = SplitOracleParam(; dcglp_param = dcglp_param,
                                                            split_index_selection_rule = LargestFractional(),
                                                            disjunctive_cut_append_rule = disjunctive_cut_append_rule,
                                                            strengthened = strengthened,
                                                            add_benders_cuts_to_master = add_benders_cuts_to_master,
                                                            fraction_of_benders_cuts_to_master = 0.05,
                                                            reuse_dcglp = reuse_dcglp,
                                                            fallback_to_typical_cuts = true,
                                                            lift = lift)

                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            typical_oracles = [ClassicalOracle(data, master; model = update_sub_model!, optimizer = optimizer); ClassicalOracle(data, master; model = update_sub_model!, optimizer = optimizer)] # for kappa & nu
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); normalization = LpDistanceNormalization(p), param = oracle_param)

                            env = BendersSeq(master, disjunctive_oracle; param = benders_param)
                            log = solve!(env)
                            @test env.termination_status == Optimal()
                            @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                            # if !isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                            #     infeasibility_report(master, x_opt, t_opt)
                            # end
                        end
                    end
                end
            end

            @testset "Unified oracle" begin
                @testset "Seq" begin
                    for strengthened in [true], add_benders_cuts_to_master in [2], reuse_dcglp in [false], p in [Inf], lift in [false], disjunctive_cut_append_rule in [AllDisjunctiveCuts()]
                        @info "solving UFLP p$i - disjunctive oracle/unified - strgthnd $strengthened; benders2master $add_benders_cuts_to_master reuse $reuse_dcglp p $p lift $lift dcut_append $disjunctive_cut_append_rule"
                        @testset "strgthnd $strengthened; benders2master $add_benders_cuts_to_master; reuse $reuse_dcglp; p $p; lift $lift; dcut_append $disjunctive_cut_append_rule" begin

                            oracle_param = SplitOracleParam(; dcglp_param = dcglp_param,
                                                            split_index_selection_rule = LargestFractional(),
                                                            disjunctive_cut_append_rule = disjunctive_cut_append_rule,
                                                            strengthened = strengthened,
                                                            add_benders_cuts_to_master = add_benders_cuts_to_master,
                                                            fraction_of_benders_cuts_to_master = 0.05,
                                                            reuse_dcglp = reuse_dcglp,
                                                            lift = lift)

                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            typical_oracles = [UnifiedOracle(data, master; model = update_sub_model!, optimizer = optimizer); UnifiedOracle(data, master; model = update_sub_model!, optimizer = optimizer)]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); normalization = LpDistanceNormalization(p), param = oracle_param)

                            env = BendersSeq(master, disjunctive_oracle; param = benders_param)
                            log = solve!(env)
                            @test env.termination_status == Optimal()
                            @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                        end
                    end
                end
            end

            @testset "Pareto oracle" begin
                @testset "Seq" begin
                    for strengthened in [true], add_benders_cuts_to_master in [2], reuse_dcglp in [false], p in [Inf], lift in [false], disjunctive_cut_append_rule in [AllDisjunctiveCuts()]
                        @info "solving UFLP p$i - disjunctive oracle/pareto - strgthnd $strengthened; benders2master $add_benders_cuts_to_master reuse $reuse_dcglp p $p lift $lift dcut_append $disjunctive_cut_append_rule"
                        @testset "strgthnd $strengthened; benders2master $add_benders_cuts_to_master; reuse $reuse_dcglp; p $p; lift $lift; dcut_append $disjunctive_cut_append_rule" begin

                            oracle_param = SplitOracleParam(; dcglp_param = dcglp_param,
                                                            split_index_selection_rule = LargestFractional(),
                                                            disjunctive_cut_append_rule = disjunctive_cut_append_rule,
                                                            strengthened = strengthened,
                                                            add_benders_cuts_to_master = add_benders_cuts_to_master,
                                                            fraction_of_benders_cuts_to_master = 0.05,
                                                            reuse_dcglp = reuse_dcglp,
                                                            lift = lift)

                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            pareto_param = ParetoOracleParam(ones(data.n_facilities))
                            typical_oracles = [ParetoOracle(data, master, pareto_param; model = update_sub_model!, optimizer = optimizer); ParetoOracle(data, master, pareto_param; model = update_sub_model!, optimizer = optimizer)]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); normalization = LpDistanceNormalization(p), param = oracle_param)

                            env = BendersSeq(master, disjunctive_oracle; param = benders_param)
                            log = solve!(env)
                            @test env.termination_status == Optimal()
                            @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                        end
                    end
                end
            end

            @testset "Classical oracle with GBC" begin
                @testset "Seq" begin
                    for strengthened in [true], add_benders_cuts_to_master in [2], reuse_dcglp in [false], p in [Inf], lift in [false], disjunctive_cut_append_rule in [AllDisjunctiveCuts()]
                        @info "solving UFLP p$i - disjunctive oracle/classical with GBC"
                        @testset "strgthnd $strengthened; benders2master $add_benders_cuts_to_master; reuse $reuse_dcglp; p $p; lift $lift; dcut_append $disjunctive_cut_append_rule" begin
                            oracle_param = SplitOracleParam(; dcglp_param = dcglp_param, split_index_selection_rule = LargestFractional(), disjunctive_cut_append_rule = disjunctive_cut_append_rule, strengthened = strengthened, add_benders_cuts_to_master = add_benders_cuts_to_master, fraction_of_benders_cuts_to_master = 0.05, reuse_dcglp = reuse_dcglp, lift = lift)
                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            typical_oracles = [ClassicalOracle(data, master; model = update_sub_gbc_model!, optimizer = optimizer); ClassicalOracle(data, master; model = update_sub_gbc_model!, optimizer = optimizer)]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); normalization = LpDistanceNormalization(p), param = oracle_param)
                            env = BendersSeq(master, disjunctive_oracle; param = benders_param)
                            log = solve!(env)
                            @test env.termination_status == Optimal()
                            @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                        end
                    end
                end
            end

            @testset "Fat knapsack oracle" begin
                function update_master_model!(model::Model, data::UFLPData)
                    optimizer = optimizer_with_attributes(CPLEX.Optimizer, "CPXPARAM_Threads" => 7, "CPX_PARAM_EPINT" => 1e-9, "CPX_PARAM_EPRHS" => 1e-9, "CPX_PARAM_EPGAP" => 1e-6, MOI.Silent() => true)
                    set_optimizer(model, optimizer)
                    @variable(model, x[1:data.n_facilities], Bin)
                    @variable(model, t[1:data.n_customers] >= -1e6)
                    @constraint(model, sum(x) >= 2)
                    @objective(model, Min, data.fixed_costs'* x + sum(t))
                    return (x = x, ), t
                end

                # fat-knapsack-based disjunctive cut has sparse gamma_t, so adding only disjunctive cut does not improve lower bound, setting add_benders_cuts_to_master = true
                @testset "Seq" begin
                    # for strengthened in [true; false], add_benders_cuts_to_master in [true; false; 2], reuse_dcglp in [true; false], p in [1.0; Inf], lift in [true; false], disjunctive_cut_append_rule in [NoDisjunctiveCuts(); AllDisjunctiveCuts(); DisjunctiveCutsSmallerIndices()]
                    for strengthened in [true], add_benders_cuts_to_master in [true], reuse_dcglp in [false], p in [Inf], lift in [false], disjunctive_cut_append_rule in [AllDisjunctiveCuts()]
                    @info "solving UFLP p$i - disjunctive oracle/fat knapsack - strgthnd $strengthened; benders2master $add_benders_cuts_to_master reuse $reuse_dcglp p $p lift $lift dcut_append $disjunctive_cut_append_rule"
                        @testset "strgthnd $strengthened; benders2master $add_benders_cuts_to_master; reuse $reuse_dcglp; p $p; lift $lift; dcut_append $disjunctive_cut_append_rule" begin

                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            typical_oracles = [UFLKnapsackOracle(data); UFLKnapsackOracle(data)] # for kappa & nu

                            oracle_param = SplitOracleParam(; dcglp_param = dcglp_param,
                                                            split_index_selection_rule = LargestFractional(),
                                                            disjunctive_cut_append_rule = disjunctive_cut_append_rule,
                                                            strengthened = strengthened,
                                                            add_benders_cuts_to_master = add_benders_cuts_to_master,
                                                            fraction_of_benders_cuts_to_master = 0.05,
                                                            reuse_dcglp = reuse_dcglp,
                                                            lift = lift)
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); normalization = LpDistanceNormalization(p), param = oracle_param)

                            env = BendersSeq(master, disjunctive_oracle; param = benders_param)
                            log = solve!(env)
                            @test env.termination_status == Optimal()
                            @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                            # if !isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                            #     infeasibility_report(master, x_opt, t_opt)
                            # end
                        end
                    end
                end
            end
        end
    end
end
