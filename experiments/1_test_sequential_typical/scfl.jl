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

    # GBC-enabled subproblem customization (y[i,j] <= x[i] via GBC)
    function update_sub_gbc_model!(model::Model, data::SCFLPData, subproblem_idx::Int; x)
        optimizer = optimizer_with_attributes(CPLEX.Optimizer, "CPXPARAM_Threads" => 7, "CPX_PARAM_EPRHS" => 1e-9, "CPX_PARAM_EPOPT" => 1e-9, "CPX_PARAM_NUMERICALEMPHASIS" => 1, MOI.Silent() => true)
        set_optimizer(model, optimizer)

        I, J = data.n_facilities, data.n_customers
        @variable(model, y[1:I, 1:J] >= 0)
        # Set objective
        cost_demands = data.costs .* data.demands[subproblem_idx]'
        @objective(model, Min, sum(cost_demands .* y))
        # Add constraints
        @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
        @constraint(model, capacity[i in 1:I], sum(data.demands[subproblem_idx][:] .* y[i,:]) <= data.capacities[i] * x[i])

        # Return GBC tuple: y[i,j] <= x[i]
        gbc_lhs = vec(y)
        gbc_rhs = [x[i] for j in 1:J for i in 1:I]  # j outer, i inner to match vec(y)
        gbc_sense = fill(UpperBound, I*J)
        return gbc_lhs, gbc_rhs, gbc_sense
    end

    for i in instances
        @testset "Instance: f25-c50-s64-r10-$i" begin

            # Load problem data
            data = read_stochastic_capacited_facility_location_problem("f25-c50-s64-r10-$i")

            # Loop parameters
            benders_param = BendersSeqParam(;
                            time_limit = 200.0,
                            gap_tolerance = 1e-6,
                            verbose = false
                        )

            instance_name = "f25-c50-s64-r10-$i"
            @assert haskey(reference_objectives, instance_name) "Missing SCFLP reference objective for $(instance_name) in $(reference_path)"
            mip_opt_val = reference_objectives[instance_name]

            @testset "Classic oracle" begin
                @info "solving SCFLP f25-c50-s64-r10-$i - classical oracle - seq..."
                master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                oracle = SeparableOracle(data, master, ClassicalOracle, data.n_scenarios; model = update_sub_model!, optimizer = optimizer)
                env = BendersSeq(master, oracle; param = benders_param)
                log = solve!(env)
                @test env.termination_status == Optimal()
                @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
            end

            @testset "Knapsack oracle" begin
                @info "solving SCFLP f25-c50-s64-r10-$i - knapsack oracle - seq..."
                master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                oracle = SeparableOracle(data, master, CFLKnapsackOracle, data.n_scenarios; model = update_sub_model!, optimizer = optimizer)
                env = BendersSeq(master, oracle; param = benders_param)
                log = solve!(env)
                @test env.termination_status == Optimal()
                @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
            end

            @testset "Unified oracle" begin
                @info "solving SCFLP f25-c50-s64-r10-$i - unified oracle - seq..."
                master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                oracle = SeparableOracle(data, master, UnifiedOracle, data.n_scenarios; model = update_sub_model!, sub_oracle_param = UnifiedOracleParam(), optimizer = optimizer)
                env = BendersSeq(master, oracle; param = benders_param)
                log = solve!(env)
                @test env.termination_status == Optimal()
                @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
            end

            @testset "Classic oracle with GBC" begin
                @info "solving SCFLP f25-c50-s64-r10-$i - classical oracle with GBC - seq..."
                master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                oracle = SeparableOracle(data, master, ClassicalOracle, data.n_scenarios; model = update_sub_gbc_model!, optimizer = optimizer)
                env = BendersSeq(master, oracle; param = benders_param)
                log = solve!(env)
                @test env.termination_status == Optimal()
                @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
            end

            @testset "Knapsack oracle with GBC" begin
                @info "solving SCFLP f25-c50-s64-r10-$i - knapsack oracle with GBC - seq..."
                master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                oracle = SeparableOracle(data, master, CFLKnapsackOracle, data.n_scenarios; model = update_sub_gbc_model!, optimizer = optimizer)
                env = BendersSeq(master, oracle; param = benders_param)
                log = solve!(env)
                @test env.termination_status == Optimal()
                @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
            end

            @testset "Pareto oracle" begin
                @info "solving SCFLP f25-c50-s64-r10-$i - pareto oracle - seq..."
                master = Master(data; model = update_master_model!, optimizer = mip_optimizer)
                param = ParetoOracleParam(fill(1.0, data.n_facilities))
                oracle = SeparableOracle(data, master, ParetoOracle, data.n_scenarios; model = update_sub_model!, sub_oracle_param = param, optimizer = optimizer)
                env = BendersSeq(master, oracle; param = benders_param)
                log = solve!(env)
                @test env.termination_status == Optimal()
                @test isapprox(mip_opt_val, env.obj_value, atol=1e-5)
            end
        end
    end
end
