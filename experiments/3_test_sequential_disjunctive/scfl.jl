using BendersX
using CSV
using DataFrames
using Test
using JuMP
using CPLEX
include(normpath(joinpath(@__DIR__, "..", "solver_defaults.jl")))

@testset verbose = true "Stochastic CFLP Sequential Benders Tests" begin
    reference_path = normpath(joinpath(@__DIR__, "..", "reference_objectives", "scflp.csv"))
    reference_df = DataFrame(CSV.File(reference_path))
    @assert nrow(reference_df) == length(unique(reference_df.instance_name)) "Duplicate SCFLP reference objectives found in $(reference_path)"
    reference_objectives = Dict(String(row.instance_name) => Float64(row.objective_value) for row in eachrow(reference_df))
    instances = 1:5

    # GBC-enabled subproblem model update (y[i,j] <= x[i] via GBC)
    function update_sub_gbc_model!(model::Model, data::SCFLPData, scen_idx::Int; x)
        optimizer = optimizer_with_attributes(CPLEX.Optimizer, "CPXPARAM_Threads" => 7, "CPX_PARAM_EPRHS" => 1e-9, "CPX_PARAM_EPOPT" => 1e-9, "CPX_PARAM_NUMERICALEMPHASIS" => 1, MOI.Silent() => true)
        set_optimizer(model, optimizer)
        I, J = data.n_facilities, data.n_customers
        @variable(model, y[1:I, 1:J] >= 0)
        cost_demands = data.costs .* data.demands[scen_idx]'
        @objective(model, Min, sum(cost_demands .* y))
        @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
        @constraint(model, capacity[i in 1:I], sum(data.demands[scen_idx][:] .* y[i,:]) <= data.capacities[i] * x[i])
        gbc_lhs = vec(y)
        gbc_rhs = [x[i] for j in 1:J for i in 1:I]
        gbc_sense = fill(UpperBound, I*J)
        return gbc_lhs, gbc_rhs, gbc_sense
    end

    for i in instances
        @testset "Instance: f25-c50-s64-r10-$i" begin
            # Load problem data
            data = read_stochastic_capacited_facility_location_problem("f25-c50-s64-r10-$i")

            # Algorithm parameters
            benders_param = BendersSeqParam(;
                                            time_limit = 2000.0,
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

            instance_name = "f25-c50-s64-r10-$i"
            @assert haskey(reference_objectives, instance_name) "Missing SCFLP reference objective for $(instance_name) in $(reference_path)"
            mip_opt_val = reference_objectives[instance_name]

            @testset "Classic oracle" begin
                @testset "Seq" begin
                    # for strengthened in [true; false], add_benders_cuts_to_master in [true; false; 2], reuse_dcglp in [true; false], p in [1.0; Inf], lift in [true; false], disjunctive_cut_append_rule in [NoDisjunctiveCuts(); AllDisjunctiveCuts(); DisjunctiveCutsSmallerIndices()]
                    for strengthened in [true], add_benders_cuts_to_master in [true], reuse_dcglp in [true], p in [1.0], lift in [true], disjunctive_cut_append_rule in [AllDisjunctiveCuts()]
                        @info "solving SCFLP f25-c50-s64-r10-$i - disjunctive oracle/classical - seq - strgthnd $strengthened; benders2master $add_benders_cuts_to_master reuse $reuse_dcglp p $p lift $lift dcut_append $disjunctive_cut_append_rule"
                        @testset "strgthnd $strengthened; benders2master $add_benders_cuts_to_master; reuse $reuse_dcglp; p $p; lift $lift; dcut_append $disjunctive_cut_append_rule" begin

                            oracle_param = SplitOracleParam(; dcglp_param = dcglp_param,
                                                                split_index_selection_rule = RandomFractional(),
                                                                disjunctive_cut_append_rule = disjunctive_cut_append_rule,
                                                                strengthened = strengthened,
                                                                add_benders_cuts_to_master = add_benders_cuts_to_master,
                                                                fraction_of_benders_cuts_to_master = 1.0,
                                                                reuse_dcglp = reuse_dcglp,
                                                                lift = lift)

                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            typical_oracle_kappa = SeparableOracle(data, master, ClassicalOracle, data.n_scenarios; model = update_sub_model!, optimizer = optimizer)
                            typical_oracle_nu = SeparableOracle(data, master, ClassicalOracle, data.n_scenarios; model = update_sub_model!, optimizer = optimizer)
                            typical_oracles = [typical_oracle_kappa; typical_oracle_nu]
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

            @testset "Knapsack oracle" begin
                @testset "Seq" begin
                    # for strengthened in [true; false], add_benders_cuts_to_master in [true; false; 2], reuse_dcglp in [true; false], p in [1.0; Inf], lift in [true; false], disjunctive_cut_append_rule in [NoDisjunctiveCuts(); AllDisjunctiveCuts(); DisjunctiveCutsSmallerIndices()]
                    for strengthened in [true], add_benders_cuts_to_master in [true], reuse_dcglp in [true], p in [1.0], lift in [true], disjunctive_cut_append_rule in [AllDisjunctiveCuts()]
                        @info "solving SCFLP f25-c50-s64-r10-$i - disjunctive oracle/classical - seq - strgthnd $strengthened; benders2master $add_benders_cuts_to_master reuse $reuse_dcglp p $p lift $lift dcut_append $disjunctive_cut_append_rule"
                        @testset "strgthnd $strengthened; benders2master $add_benders_cuts_to_master; reuse $reuse_dcglp; p $p; lift $lift; dcut_append $disjunctive_cut_append_rule" begin

                            oracle_param = SplitOracleParam(; dcglp_param = dcglp_param,
                                                                split_index_selection_rule = RandomFractional(),
                                                                disjunctive_cut_append_rule = disjunctive_cut_append_rule,
                                                                strengthened = strengthened,
                                                                add_benders_cuts_to_master = add_benders_cuts_to_master,
                                                                fraction_of_benders_cuts_to_master = 1.0,
                                                                reuse_dcglp = reuse_dcglp,
                                                                lift = lift)

                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            typical_oracle_kappa = SeparableOracle(data, master, CFLKnapsackOracle, data.n_scenarios; model = update_sub_model!, optimizer = optimizer)
                            typical_oracle_nu = SeparableOracle(data, master, CFLKnapsackOracle, data.n_scenarios; model = update_sub_model!, optimizer = optimizer)
                            typical_oracles = [typical_oracle_kappa; typical_oracle_nu]
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
                    for strengthened in [true], add_benders_cuts_to_master in [true], reuse_dcglp in [true], p in [1.0], lift in [true], disjunctive_cut_append_rule in [AllDisjunctiveCuts()]
                        @info "solving SCFLP f25-c50-s64-r10-$i - disjunctive oracle/unified - seq - strgthnd $strengthened; benders2master $add_benders_cuts_to_master reuse $reuse_dcglp p $p lift $lift dcut_append $disjunctive_cut_append_rule"
                        @testset "strgthnd $strengthened; benders2master $add_benders_cuts_to_master; reuse $reuse_dcglp; p $p; lift $lift; dcut_append $disjunctive_cut_append_rule" begin

                            oracle_param = SplitOracleParam(; dcglp_param = dcglp_param,
                                                                split_index_selection_rule = RandomFractional(),
                                                                disjunctive_cut_append_rule = disjunctive_cut_append_rule,
                                                                strengthened = strengthened,
                                                                add_benders_cuts_to_master = add_benders_cuts_to_master,
                                                                fraction_of_benders_cuts_to_master = 1.0,
                                                                reuse_dcglp = reuse_dcglp,
                                                                lift = lift)

                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            typical_oracle_kappa = SeparableOracle(data, master, UnifiedOracle, data.n_scenarios; model = update_sub_model!, sub_oracle_param = UnifiedOracleParam(), optimizer = optimizer)
                            typical_oracle_nu = SeparableOracle(data, master, UnifiedOracle, data.n_scenarios; model = update_sub_model!, sub_oracle_param = UnifiedOracleParam(), optimizer = optimizer)
                            typical_oracles = [typical_oracle_kappa; typical_oracle_nu]
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
                    for strengthened in [true], add_benders_cuts_to_master in [true], reuse_dcglp in [true], p in [1.0], lift in [true], disjunctive_cut_append_rule in [AllDisjunctiveCuts()]
                        @info "solving SCFLP f25-c50-s64-r10-$i - disjunctive oracle/pareto - seq - strgthnd $strengthened; benders2master $add_benders_cuts_to_master reuse $reuse_dcglp p $p lift $lift dcut_append $disjunctive_cut_append_rule"
                        @testset "strgthnd $strengthened; benders2master $add_benders_cuts_to_master; reuse $reuse_dcglp; p $p; lift $lift; dcut_append $disjunctive_cut_append_rule" begin

                            oracle_param = SplitOracleParam(; dcglp_param = dcglp_param,
                                                                split_index_selection_rule = RandomFractional(),
                                                                disjunctive_cut_append_rule = disjunctive_cut_append_rule,
                                                                strengthened = strengthened,
                                                                add_benders_cuts_to_master = add_benders_cuts_to_master,
                                                                fraction_of_benders_cuts_to_master = 1.0,
                                                                reuse_dcglp = reuse_dcglp,
                                                                lift = lift)

                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            pareto_param = ParetoOracleParam(ones(data.n_facilities))
                            typical_oracle_kappa = SeparableOracle(data, master, ParetoOracle, data.n_scenarios; model = update_sub_model!, sub_oracle_param = pareto_param, optimizer = optimizer)
                            typical_oracle_nu = SeparableOracle(data, master, ParetoOracle, data.n_scenarios; model = update_sub_model!, sub_oracle_param = pareto_param, optimizer = optimizer)
                            typical_oracles = [typical_oracle_kappa; typical_oracle_nu]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); normalization = LpDistanceNormalization(p), param = oracle_param)
                            env = BendersSeq(master, disjunctive_oracle; param = benders_param)

                            log = solve!(env)
                            @test env.termination_status == Optimal()
                            @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                        end
                    end
                end
            end

            @testset "Classic oracle with GBC" begin
                @testset "Seq" begin
                    for strengthened in [true], add_benders_cuts_to_master in [true], reuse_dcglp in [true], p in [1.0], lift in [true], disjunctive_cut_append_rule in [AllDisjunctiveCuts()]
                        @info "solving SCFLP f25-c50-s64-r10-$i - disjunctive oracle/classical with GBC - seq"
                        @testset "strgthnd $strengthened; benders2master $add_benders_cuts_to_master; reuse $reuse_dcglp; p $p; lift $lift; dcut_append $disjunctive_cut_append_rule" begin
                            oracle_param = SplitOracleParam(; dcglp_param = dcglp_param, split_index_selection_rule = RandomFractional(), disjunctive_cut_append_rule = disjunctive_cut_append_rule, strengthened = strengthened, add_benders_cuts_to_master = add_benders_cuts_to_master, fraction_of_benders_cuts_to_master = 1.0, reuse_dcglp = reuse_dcglp, lift = lift)
                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            typical_oracle_kappa = SeparableOracle(data, master, ClassicalOracle, data.n_scenarios; model = update_sub_gbc_model!, optimizer = optimizer)
                            typical_oracle_nu = SeparableOracle(data, master, ClassicalOracle, data.n_scenarios; model = update_sub_gbc_model!, optimizer = optimizer)
                            typical_oracles = [typical_oracle_kappa; typical_oracle_nu]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); normalization = LpDistanceNormalization(p), param = oracle_param)
                            env = BendersSeq(master, disjunctive_oracle; param = benders_param)
                            log = solve!(env)
                            @test env.termination_status == Optimal()
                            @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
                        end
                    end
                end
            end

            @testset "Knapsack oracle with GBC" begin
                @testset "Seq" begin
                    for strengthened in [true], add_benders_cuts_to_master in [true], reuse_dcglp in [true], p in [1.0], lift in [true], disjunctive_cut_append_rule in [AllDisjunctiveCuts()]
                        @info "solving SCFLP f25-c50-s64-r10-$i - disjunctive oracle/knapsack with GBC - seq"
                        @testset "strgthnd $strengthened; benders2master $add_benders_cuts_to_master; reuse $reuse_dcglp; p $p; lift $lift; dcut_append $disjunctive_cut_append_rule" begin
                            oracle_param = SplitOracleParam(; dcglp_param = dcglp_param, split_index_selection_rule = RandomFractional(), disjunctive_cut_append_rule = disjunctive_cut_append_rule, strengthened = strengthened, add_benders_cuts_to_master = add_benders_cuts_to_master, fraction_of_benders_cuts_to_master = 1.0, reuse_dcglp = reuse_dcglp, lift = lift)
                            master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                            typical_oracle_kappa = SeparableOracle(data, master, CFLKnapsackOracle, data.n_scenarios; model = update_sub_gbc_model!, optimizer = optimizer)
                            typical_oracle_nu = SeparableOracle(data, master, CFLKnapsackOracle, data.n_scenarios; model = update_sub_gbc_model!, optimizer = optimizer)
                            typical_oracles = [typical_oracle_kappa; typical_oracle_nu]
                            disjunctive_oracle = SplitOracle(master, Tuple(typical_oracles); normalization = LpDistanceNormalization(p), param = oracle_param)
                            env = BendersSeq(master, disjunctive_oracle; param = benders_param)
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
