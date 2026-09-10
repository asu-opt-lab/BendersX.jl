"""
GBC (Generalized Bound Constraints) Test Suite

Tests various scenarios for GBC support:
1. Basic upper bound constraint: y <= x
2. Lower bound constraint: y >= 0.1*x
3. Fixing constraint: y == x
4. Scaled single variable: y <= 2*x
5. Multi-variable affine combination: y <= 0.5*x[1] + 0.5*x[2]
6. Partial y constraints: only subset of y has GBC
7. Mixed bound types: some upper, some lower, some fixed
8. No GBC (`nothing` returned)
9. Mixed RHS types: mixing VariableRef and AffExpr for optimization testing
10. Error handling: DimensionMismatch, ArgumentError, Warning on empty LHS

Uses a simple transportation-like problem for testing.
"""

using BendersX
using Test
using JuMP
using CPLEX

# =============================================================================
# Test Problem: Simple Assignment Problem
# Minimize: sum(c[i,j] * y[i,j])
# Subject to: sum(y[:,j]) == 1 for all j (demand satisfied)
#            y[i,j] >= 0
#            y[i,j] <= x[i] for all i,j (facility open constraint - GBC)
# =============================================================================

struct SimpleAssignmentData
    n_facilities::Int
    n_customers::Int
    costs::Matrix{Float64}
end

function create_test_data(n_facilities::Int=3, n_customers::Int=4)
    costs = rand(n_facilities, n_customers) .* 10
    return SimpleAssignmentData(n_facilities, n_customers, costs)
end

function update_master_simple_model!(model::Model, data::SimpleAssignmentData)
    optimizer = optimizer_with_attributes(
        CPLEX.Optimizer, "CPXPARAM_Threads" => 1, MOI.Silent() => true)
    set_optimizer(model, optimizer)
    
    @variable(model, x[1:data.n_facilities], Bin)
    @variable(model, t >= -1e6)
    @objective(model, Min, t)
    
    # At least one facility must be open
    @constraint(model, sum(x) >= 1)
    
    return (x = x, ), t
end

function solve_mip_reference(data::SimpleAssignmentData)
    model = Model()
    optimizer = optimizer_with_attributes(
        CPLEX.Optimizer, "CPXPARAM_Threads" => 1, MOI.Silent() => true)
    set_optimizer(model, optimizer)
    
    I, J = data.n_facilities, data.n_customers
    @variable(model, x[1:I], Bin)
    @variable(model, y[1:I, 1:J] >= 0)
    
    @objective(model, Min, sum(data.costs .* y))
    @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
    @constraint(model, facility_open[i in 1:I, j in 1:J], y[i,j] <= x[i])
    @constraint(model, sum(x) >= 1)
    
    optimize!(model)
    return objective_value(model)
end

# =============================================================================
# Test Scenarios
# =============================================================================

@testset verbose = true "GBC Test Suite" begin
    
    @testset "Scenario 1: Basic Upper Bound" begin
        # Tests basic upper bound constraint y <= x
        data = create_test_data()
        mip_opt = solve_mip_reference(data)
        
        function update_sub_upper_model!(model::Model, data::SimpleAssignmentData, scen_idx::Int; x)
            optimizer = optimizer_with_attributes(
                CPLEX.Optimizer, "CPXPARAM_Threads" => 1, MOI.Silent() => true)
            set_optimizer(model, optimizer)
            
            I, J = data.n_facilities, data.n_customers
            @variable(model, y[1:I, 1:J] >= 0)
            
            @objective(model, Min, sum(data.costs .* y))
            @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
            
            gbc_lhs = vec(y)
            gbc_rhs = [x[i] for i in 1:I for j in 1:J]
            gbc_sense = fill(UpperBound, I*J)
            
            return gbc_lhs, gbc_rhs, gbc_sense
        end
        
        benders_param = BendersSeqParam(time_limit = 60.0, gap_tolerance = 1e-6, verbose = false)
        master = Master(data; model = update_master_simple_model!)
        oracle = ClassicalOracle(data, master; model = update_sub_upper_model!)
        env = BendersSeq(master, oracle; param = benders_param)
        solve!(env)
        
        @test env.termination_status == Optimal()
        @test isapprox(mip_opt, env.obj_value, atol=1e-4)
    end
    
    @testset "Scenario 2: Lower Bound Constraint" begin
        # Tests y >= x constraint (facility must serve if open)
        # Modified problem: if facility is open, it must serve at least some amount
        data = create_test_data()

        function solve_mip_scenario_3(data::SimpleAssignmentData)
            model = Model()
            optimizer = optimizer_with_attributes(
                CPLEX.Optimizer, "CPXPARAM_Threads" => 1, MOI.Silent() => true)
            set_optimizer(model, optimizer)

            I, J = data.n_facilities, data.n_customers
            @variable(model, x[1:I], Bin)
            @variable(model, y[1:I, 1:J] >= 0)
            @variable(model, z[1:I] >= 0)  # Total service from facility i

            @objective(model, Min, sum(data.costs .* y))
            @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
            @constraint(model, facility_open[i in 1:I, j in 1:J], y[i,j] <= 1)
            @constraint(model, total_service[i in 1:I], z[i] == sum(y[i,:]))
            @constraint(model, min_service[i in 1:I], z[i] >= 0.1 * x[i])  # GBC: Lower bound
            @constraint(model, sum(x) >= 1)

            optimize!(model)
            return objective_value(model)
        end

        mip_opt = solve_mip_scenario_3(data)
        
        function update_sub_lower_model!(model::Model, data::SimpleAssignmentData, scen_idx::Int; x)
            optimizer = optimizer_with_attributes(
                CPLEX.Optimizer, "CPXPARAM_Threads" => 1, MOI.Silent() => true)
            set_optimizer(model, optimizer)
            
            I, J = data.n_facilities, data.n_customers
            @variable(model, y[1:I, 1:J] >= 0)
            @variable(model, z[1:I] >= 0)  # Total service from facility i
            
            @objective(model, Min, sum(data.costs .* y))
            @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
            @constraint(model, facility_open[i in 1:I, j in 1:J], y[i,j] <= 1)  # Upper bound
            @constraint(model, total_service[i in 1:I], z[i] == sum(y[i,:]))
            
            # Lower bound GBC: z[i] >= 0.1 * x[i] (if open, serve at least 10%)
            gbc_lhs = collect(z)
            gbc_rhs = [x[i] for i in 1:I]
            gbc_sense = fill(LowerBound, I)
            
            return gbc_lhs, gbc_rhs, gbc_sense
        end
        
        benders_param = BendersSeqParam(time_limit = 60.0, gap_tolerance = 1e-6, verbose = false)
        master = Master(data; model = update_master_simple_model!)
        oracle = ClassicalOracle(data, master; model = update_sub_lower_model!)
        env = BendersSeq(master, oracle; param = benders_param)
        solve!(env)

        @test env.termination_status == Optimal()
        @test isapprox(mip_opt, env.obj_value, atol=1e-4)
    end

    @testset "Scenario 3: Fixing Constraint" begin
        # Tests y == x constraint
        data = create_test_data()

        function solve_mip_scenario_4(data::SimpleAssignmentData)
            model = Model()
            optimizer = optimizer_with_attributes(
                CPLEX.Optimizer, "CPXPARAM_Threads" => 1, MOI.Silent() => true)
            set_optimizer(model, optimizer)

            I, J = data.n_facilities, data.n_customers
            @variable(model, x[1:I], Bin)
            @variable(model, y[1:I, 1:J] >= 0)
            @variable(model, w[1:I] >= 0)  # Auxiliary: exactly equals x

            @objective(model, Min, sum(data.costs .* y))
            @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
            @constraint(model, facility_open[i in 1:I, j in 1:J], y[i,j] <= w[i])
            @constraint(model, fixed_capacity[i in 1:I], w[i] == x[i])  # GBC: Fixed bound
            @constraint(model, sum(x) >= 1)

            optimize!(model)
            return objective_value(model)
        end

        mip_opt = solve_mip_scenario_4(data)
        
        function update_sub_fixed_model!(model::Model, data::SimpleAssignmentData, scen_idx::Int; x)
            optimizer = optimizer_with_attributes(
                CPLEX.Optimizer, "CPXPARAM_Threads" => 1, MOI.Silent() => true)
            set_optimizer(model, optimizer)
            
            I, J = data.n_facilities, data.n_customers
            @variable(model, y[1:I, 1:J] >= 0)
            @variable(model, w[1:I] >= 0)  # Auxiliary: exactly equals x
            
            @objective(model, Min, sum(data.costs .* y))
            @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
            @constraint(model, facility_open[i in 1:I, j in 1:J], y[i,j] <= w[i])
            
            # Fixed bound GBC: w[i] == x[i]
            gbc_lhs = collect(w)
            gbc_rhs = [x[i] for i in 1:I]
            gbc_sense = fill(Fixed, I)
            
            return gbc_lhs, gbc_rhs, gbc_sense
        end
        
        benders_param = BendersSeqParam(time_limit = 60.0, gap_tolerance = 1e-6, verbose = false)
        master = Master(data; model = update_master_simple_model!)
        oracle = ClassicalOracle(data, master; model = update_sub_fixed_model!)
        env = BendersSeq(master, oracle; param = benders_param)
        solve!(env)

        @test env.termination_status == Optimal()
        @test isapprox(mip_opt, env.obj_value, atol=1e-4)
    end

    @testset "Scenario 4: Scaled Single Variable" begin
        # Tests y <= 2*x constraint
        data = create_test_data()

        function solve_mip_scenario_5(data::SimpleAssignmentData)
            model = Model()
            optimizer = optimizer_with_attributes(
                CPLEX.Optimizer, "CPXPARAM_Threads" => 1, MOI.Silent() => true)
            set_optimizer(model, optimizer)

            I, J = data.n_facilities, data.n_customers
            @variable(model, x[1:I], Bin)
            @variable(model, y[1:I, 1:J] >= 0)

            @objective(model, Min, sum(data.costs .* y))
            @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
            @constraint(model, facility_open[i in 1:I, j in 1:J], y[i,j] <= 2.0 * x[i])  # GBC: Scaled (2.0)
            @constraint(model, sum(x) >= 1)

            optimize!(model)
            return objective_value(model)
        end

        mip_opt = solve_mip_scenario_5(data)
        
        function update_sub_scaled_model!(model::Model, data::SimpleAssignmentData, scen_idx::Int; x)
            optimizer = optimizer_with_attributes(
                CPLEX.Optimizer, "CPXPARAM_Threads" => 1, MOI.Silent() => true)
            set_optimizer(model, optimizer)
            
            I, J = data.n_facilities, data.n_customers
            @variable(model, y[1:I, 1:J] >= 0)
            
            @objective(model, Min, sum(data.costs .* y))
            @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
            
            # Scaled GBC: y[i,j] <= 2*x[i] (allow up to 2 units per facility)
            gbc_lhs = vec(y)
            gbc_rhs = [2.0 * x[i] for i in 1:I for j in 1:J]
            gbc_sense = fill(UpperBound, I*J)
            
            return gbc_lhs, gbc_rhs, gbc_sense
        end
        
        benders_param = BendersSeqParam(time_limit = 60.0, gap_tolerance = 1e-6, verbose = false)
        master = Master(data; model = update_master_simple_model!)
        oracle = ClassicalOracle(data, master; model = update_sub_scaled_model!)
        env = BendersSeq(master, oracle; param = benders_param)
        solve!(env)

        @test env.termination_status == Optimal()
        @test isapprox(mip_opt, env.obj_value, atol=1e-4)
    end

    @testset "Scenario 5: Multi-Variable Affine Combination" begin
        # Tests y <= x[1] + x[2] constraint (depends on multiple x)
        data = create_test_data(4, 3)  # 4 facilities, 3 customers

        function solve_mip_scenario_6(data::SimpleAssignmentData)
            model = Model()
            optimizer = optimizer_with_attributes(
                CPLEX.Optimizer, "CPXPARAM_Threads" => 1, MOI.Silent() => true)
            set_optimizer(model, optimizer)

            I, J = data.n_facilities, data.n_customers
            @variable(model, x[1:I], Bin)
            @variable(model, y[1:J] >= 0)  # Total service to each customer

            @objective(model, Min, sum(data.costs[1,:] .* y))  # Use costs from facility 1
            @constraint(model, demand[j in 1:J], y[j] <= 1)
            @constraint(model, total, sum(y) >= 1)
            @constraint(model, affine_gbc[j in 1:J], y[j] <= 0.5*x[1] + 0.5*x[2])  # GBC: Affine combination
            @constraint(model, sum(x) >= 2)  # At least 2 facilities

            optimize!(model)
            return objective_value(model)
        end

        mip_opt = solve_mip_scenario_6(data)

        function update_master_pairs_model!(model::Model, data::SimpleAssignmentData)
            optimizer = optimizer_with_attributes(
                CPLEX.Optimizer, "CPXPARAM_Threads" => 1, MOI.Silent() => true)
            set_optimizer(model, optimizer)
            
            @variable(model, x[1:data.n_facilities], Bin)
            @variable(model, t >= -1e6)
            @objective(model, Min, t)
            @constraint(model, sum(x) >= 2)  # At least 2 facilities
            
            return (x = x, ), t
        end
        
        function update_sub_affine_model!(model::Model, data::SimpleAssignmentData, scen_idx::Int; x)
            optimizer = optimizer_with_attributes(
                CPLEX.Optimizer, "CPXPARAM_Threads" => 1, MOI.Silent() => true)
            set_optimizer(model, optimizer)
            
            I, J = data.n_facilities, data.n_customers
            @variable(model, y[1:J] >= 0)  # Total service to each customer
            
            @objective(model, Min, sum(data.costs[1,:] .* y))  # Use costs from facility 1
            @constraint(model, demand[j in 1:J], y[j] <= 1)
            @constraint(model, total, sum(y) >= 1)
            
            # Affine GBC: y[j] <= x[1] + x[2] for each customer j
            # This means customer j can be served only if facilities 1 OR 2 are open
            gbc_lhs = collect(y)
            gbc_rhs = [0.5*x[1] + 0.5*x[2] for _ in 1:J]  # y[j] <= 0.5*x[1] + 0.5*x[2]
            gbc_sense = fill(UpperBound, J)
            
            return gbc_lhs, gbc_rhs, gbc_sense
        end
        
        benders_param = BendersSeqParam(time_limit = 60.0, gap_tolerance = 1e-6, verbose = false)
        master = Master(data; model = update_master_pairs_model!)
        oracle = ClassicalOracle(data, master; model = update_sub_affine_model!)
        env = BendersSeq(master, oracle; param = benders_param)
        solve!(env)

        @test env.termination_status == Optimal()
        @test isapprox(mip_opt, env.obj_value, atol=1e-4)
    end

    @testset "Scenario 6: Partial Y Constraints" begin
        # Tests case where only subset of y has GBC constraints
        data = create_test_data()

        function solve_mip_scenario_7(data::SimpleAssignmentData)
            model = Model()
            optimizer = optimizer_with_attributes(
                CPLEX.Optimizer, "CPXPARAM_Threads" => 1, MOI.Silent() => true)
            set_optimizer(model, optimizer)

            I, J = data.n_facilities, data.n_customers
            @variable(model, x[1:I], Bin)
            @variable(model, y[1:I, 1:J] >= 0)

            @objective(model, Min, sum(data.costs .* y))
            @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
            # GBC: Only first facility has constraint
            @constraint(model, facility_1_open[j in 1:J], y[1,j] <= x[1])
            # No constraints on y[2:I, :] with respect to x
            @constraint(model, sum(x) >= 1)

            optimize!(model)
            return objective_value(model)
        end

        mip_opt = solve_mip_scenario_7(data)
        
        function update_sub_partial_model!(model::Model, data::SimpleAssignmentData, scen_idx::Int; x)
            optimizer = optimizer_with_attributes(
                CPLEX.Optimizer, "CPXPARAM_Threads" => 1, MOI.Silent() => true)
            set_optimizer(model, optimizer)
            
            I, J = data.n_facilities, data.n_customers
            @variable(model, y[1:I, 1:J] >= 0)
            
            @objective(model, Min, sum(data.costs .* y))
            @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
            
            # Only first facility has GBC constraint
            # y[1,j] <= x[1] for all j
            gbc_lhs = [y[1,j] for j in 1:J]
            gbc_rhs = [x[1] for _ in 1:J]
            gbc_sense = fill(UpperBound, J)
            
            return gbc_lhs, gbc_rhs, gbc_sense
        end
        
        benders_param = BendersSeqParam(time_limit = 60.0, gap_tolerance = 1e-6, verbose = false)
        master = Master(data; model = update_master_simple_model!)
        oracle = ClassicalOracle(data, master; model = update_sub_partial_model!)
        env = BendersSeq(master, oracle; param = benders_param)
        solve!(env)

        @test env.termination_status == Optimal()
        @test isapprox(mip_opt, env.obj_value, atol=1e-4)
    end

    @testset "Scenario 7: Mixed Bound Types" begin
        # Tests case with different bound types for different variables
        data = create_test_data()

        function solve_mip_scenario_8(data::SimpleAssignmentData)
            model = Model()
            optimizer = optimizer_with_attributes(
                CPLEX.Optimizer, "CPXPARAM_Threads" => 1, MOI.Silent() => true)
            set_optimizer(model, optimizer)

            I, J = data.n_facilities, data.n_customers
            @variable(model, x[1:I], Bin)
            @variable(model, y[1:I, 1:J] >= 0)
            @variable(model, z[1:I] >= 0)  # Total service
            @variable(model, w[1:I] >= 0)  # Fixed capacity

            @objective(model, Min, sum(data.costs .* y))
            @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
            @constraint(model, facility_open[i in 1:I, j in 1:J], y[i,j] <= w[i])
            @constraint(model, total_service[i in 1:I], z[i] == sum(y[i,:]))
            # Mixed GBC:
            @constraint(model, fixed_capacity[i in 1:I], w[i] == x[i])  # Fixed bound
            @constraint(model, min_service[i in 1:I], z[i] >= 0.01 * x[i])  # Lower bound
            @constraint(model, sum(x) >= 1)

            optimize!(model)
            return objective_value(model)
        end

        mip_opt = solve_mip_scenario_8(data)
        
        function update_sub_mixed_model!(model::Model, data::SimpleAssignmentData, scen_idx::Int; x)
            optimizer = optimizer_with_attributes(
                CPLEX.Optimizer, "CPXPARAM_Threads" => 1, MOI.Silent() => true)
            set_optimizer(model, optimizer)
            
            I, J = data.n_facilities, data.n_customers
            @variable(model, y[1:I, 1:J] >= 0)
            @variable(model, z[1:I] >= 0)  # Total service
            @variable(model, w[1:I] >= 0)  # Fixed capacity
            
            @objective(model, Min, sum(data.costs .* y))
            @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
            @constraint(model, facility_open[i in 1:I, j in 1:J], y[i,j] <= w[i])
            @constraint(model, total_service[i in 1:I], z[i] == sum(y[i,:]))
            
            # Mixed GBC:
            # - w[i] == x[i] (Fixed)
            # - z[i] >= 0.01*x[i] (Lower, if open serve at least 1%)
            gbc_lhs = vcat(collect(w), collect(z))
            gbc_rhs = vcat([x[i] for i in 1:I], [0.01 * x[i] for i in 1:I])
            gbc_sense = vcat(fill(Fixed, I), fill(LowerBound, I))
            
            return gbc_lhs, gbc_rhs, gbc_sense
        end
        
        benders_param = BendersSeqParam(time_limit = 60.0, gap_tolerance = 1e-6, verbose = false)
        master = Master(data; model = update_master_simple_model!)
        oracle = ClassicalOracle(data, master; model = update_sub_mixed_model!)
        env = BendersSeq(master, oracle; param = benders_param)
        solve!(env)

        @test env.termination_status == Optimal()
        @test isapprox(mip_opt, env.obj_value, atol=1e-4)
    end

    @testset "Scenario 8: No GBC (Nothing Returned)" begin
        # Tests that code works when the model-update function returns nothing
        data = create_test_data()
        mip_opt = solve_mip_reference(data)
        
        function update_sub_no_gbc_model!(model::Model, data::SimpleAssignmentData, scen_idx::Int; x)
            optimizer = optimizer_with_attributes(
                CPLEX.Optimizer, "CPXPARAM_Threads" => 1, MOI.Silent() => true)
            set_optimizer(model, optimizer)
            
            I, J = data.n_facilities, data.n_customers
            @variable(model, y[1:I, 1:J] >= 0)
            
            @objective(model, Min, sum(data.costs .* y))
            @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
            @constraint(model, facility_open[i in 1:I, j in 1:J], y[i,j] <= x[i])  # Regular constraint
            
            # No GBC return
            return nothing
        end
        
        benders_param = BendersSeqParam(time_limit = 60.0, gap_tolerance = 1e-6, verbose = false)
        master = Master(data; model = update_master_simple_model!)
        oracle = ClassicalOracle(data, master; model = update_sub_no_gbc_model!)
        env = BendersSeq(master, oracle; param = benders_param)
        solve!(env)

        @test env.termination_status == Optimal()
        @test isapprox(mip_opt, env.obj_value, atol=1e-4)
    end

    @testset "Scenario 9: Mixed RHS Types (VariableRef + AffExpr)" begin
        # Tests mixing VariableRef and AffExpr in gbc_rhs
        data = create_test_data()
        mip_opt = solve_mip_reference(data)
        
        function update_sub_union_model!(model::Model, data::SimpleAssignmentData, scen_idx::Int; x)
            optimizer = optimizer_with_attributes(
                CPLEX.Optimizer, "CPXPARAM_Threads" => 1, MOI.Silent() => true)
            set_optimizer(model, optimizer)
            
            I, J = data.n_facilities, data.n_customers
            @variable(model, y[1:I, 1:J] >= 0)
            
            @objective(model, Min, sum(data.costs .* y))
            @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
            
            # Use mixed Vector{Union{VariableRef, AffExpr}}
            # First half: use VariableRef
            # Second half: use AffExpr (1.0 * x)
            gbc_lhs = vec(y)
            gbc_rhs = Vector{Union{VariableRef, AffExpr}}(undef, length(y))
            
            idx = 1
            mid = length(y) ÷ 2
            for i in 1:I, j in 1:J
                if idx <= mid
                     gbc_rhs[idx] = x[i]  # VariableRef
                else
                     gbc_rhs[idx] = 1.0 * x[i] + 0.0  # AffExpr
                end
                idx += 1
            end
            
            gbc_sense = fill(UpperBound, I*J)
            return gbc_lhs, gbc_rhs, gbc_sense
        end
        
        benders_param = BendersSeqParam(time_limit = 60.0, gap_tolerance = 1e-6, verbose = false)
        master = Master(data; model = update_master_simple_model!)
        oracle = ClassicalOracle(data, master; model = update_sub_union_model!)
        env = BendersSeq(master, oracle; param = benders_param)
        solve!(env)
        
        @test env.termination_status == Optimal()
        @test isapprox(mip_opt, env.obj_value, atol=1e-4)
    end
    
    @testset "Scenario 10: Error Handling" begin
        data = create_test_data()
        
        # 1. Test DimensionMismatch
        function update_sub_dim_error_model!(model::Model, data::SimpleAssignmentData, scen_idx::Int; x)
            gbc_lhs = [VariableRef(model, MOI.VariableIndex(1))]
            gbc_rhs = union_type([x[1], x[1]]) # Length 2
            gbc_sense = [UpperBound]
            return gbc_lhs, gbc_rhs, gbc_sense
        end
        
        master = Master(data; model = update_master_simple_model!)
        # Helper function to supply Union type
        union_type(v) = Vector{Union{VariableRef, AffExpr}}(v)

        @test_throws DimensionMismatch ClassicalOracle(data, master; model = update_sub_dim_error_model!)
        
        # 2. Test ArgumentError (Invalid tuple length)
        function update_sub_arg_error_model!(model::Model, data::SimpleAssignmentData, scen_idx::Int; x)
            return ([x[1]], [x[1]]) # Length 2, expected 3
        end
        @test_throws ArgumentError ClassicalOracle(data, master; model = update_sub_arg_error_model!)

        # 3. Test Empty LHS Warning
        function update_sub_empty_warn_model!(model::Model, data::SimpleAssignmentData, scen_idx::Int; x)
            return VariableRef[], Union{VariableRef, AffExpr}[], GBCBoundType[]
        end
        
        # Capture warning
        @test_logs (:warn, "GBC tuple returned but gbc_lhs is empty. No GBC constraints will be applied.") begin
            ClassicalOracle(data, master; model = update_sub_empty_warn_model!)
        end

        # 4. Test ArgumentError: gbc_lhs contains a master variable (from x)
        function update_sub_lhs_master_error_model!(model::Model, data::SimpleAssignmentData, scen_idx::Int; x)
            optimizer = optimizer_with_attributes(
                CPLEX.Optimizer, "CPXPARAM_Threads" => 1, MOI.Silent() => true)
            set_optimizer(model, optimizer)
            
            I, J = data.n_facilities, data.n_customers
            @variable(model, y[1:I, 1:J] >= 0)
            @objective(model, Min, sum(data.costs .* y))
            
            # ERROR: gbc_lhs contains x[1] (a master variable), should contain subproblem var
            gbc_lhs = [x[1]]
            gbc_rhs = [x[1]]
            gbc_sense = [UpperBound]
            return gbc_lhs, gbc_rhs, gbc_sense
        end

        err = try
            ClassicalOracle(data, master; model = update_sub_lhs_master_error_model!)
            nothing
        catch e
            e
        end

        @test err isa ArgumentError
        @test occursin("should only contain a single subproblem variable", err.msg)

        # 5. Test ArgumentError: gbc_rhs contains a non-master variable (subproblem variable)
        function update_sub_rhs_nonmaster_error_model!(model::Model, data::SimpleAssignmentData, scen_idx::Int; x)
            optimizer = optimizer_with_attributes(
                CPLEX.Optimizer, "CPXPARAM_Threads" => 1, MOI.Silent() => true)
            set_optimizer(model, optimizer)
            
            I, J = data.n_facilities, data.n_customers
            @variable(model, y[1:I, 1:J] >= 0)
            @variable(model, w[1:I] >= 0)
            @objective(model, Min, sum(data.costs .* y))
            
            # ERROR: gbc_rhs contains y[1,1] (a subproblem variable), should contain master var
            gbc_lhs = [w[1]]
            gbc_rhs = [y[1, 1]]  # y is a subproblem variable, not a master variable!
            gbc_sense = [UpperBound]
            return gbc_lhs, gbc_rhs, gbc_sense
        end

        err = try
            ClassicalOracle(data, master; model = update_sub_rhs_nonmaster_error_model!)
            nothing
        catch e
            e
        end

        @test err isa ArgumentError
        @test occursin("not a copied master variable", err.msg)


        # 6. Test ArgumentError: gbc_lhs contains a non-VariableRef element.
        function update_sub_lhs_error_model!(model::Model, data::SimpleAssignmentData, scen_idx::Int; x)
            optimizer = optimizer_with_attributes(
                CPLEX.Optimizer, "CPXPARAM_Threads" => 1, MOI.Silent() => true)
            set_optimizer(model, optimizer)
            
            I, J = data.n_facilities, data.n_customers
            @variable(model, y[1:I, 1:J] >= 0)
            @variable(model, w[1:I] >= 0)
            @objective(model, Min, sum(data.costs .* y))
            
            # ERROR: gbc_lhs contains an AffExpr (y[1,1] + y[1,2]), should only contain VariableRef
            gbc_lhs = [y[1,1] + y[1,2]; y[1]]
            gbc_rhs = [x[1]; x[2]]
            gbc_sense = [UpperBound; Fixed]
            return gbc_lhs, gbc_rhs, gbc_sense
        end
        
        err = try
            ClassicalOracle(data, master; model = update_sub_lhs_error_model!)
            nothing
        catch e
            e
        end

        @test err isa ArgumentError
        @test occursin("not a VariableRef", err.msg)

        # 7. Test ArgumentError: gbc_rhs contains a non-affine expression.
        function update_sub_rhs_nonaffine_error_model!(model::Model, data::SimpleAssignmentData, scen_idx::Int; x)
            optimizer = optimizer_with_attributes(
                CPLEX.Optimizer, "CPXPARAM_Threads" => 1, MOI.Silent() => true)
            set_optimizer(model, optimizer)
            
            I, J = data.n_facilities, data.n_customers
            @variable(model, y[1:I, 1:J] >= 0)
            @variable(model, w[1:I] >= 0)
            @objective(model, Min, sum(data.costs .* y))
            
            # ERROR: gbc_rhs contains x[1]^2 + x[2] (a quadratic expression), should only contain affine expressions
            gbc_lhs = [y[1,1]; y[1,2]]
            gbc_rhs = [x[1]^2 + x[2]; x[2]]
            gbc_sense = [UpperBound; Fixed]
            return gbc_lhs, gbc_rhs, gbc_sense
        end
        
        err = try
            ClassicalOracle(data, master; model = update_sub_rhs_nonaffine_error_model!)
            nothing
        catch e
            e
        end

        @test err isa ArgumentError
        @test occursin("contains a non-affine expression", err.msg)

        # 8. Test ArgumentError: gbc_sense contains not acceptable value
        function update_sub_sense_error_model!(model::Model, data::SimpleAssignmentData, scen_idx::Int; x)
            optimizer = optimizer_with_attributes(
                CPLEX.Optimizer, "CPXPARAM_Threads" => 1, MOI.Silent() => true)
            set_optimizer(model, optimizer)
            
            I, J = data.n_facilities, data.n_customers
            @variable(model, y[1:I, 1:J] >= 0)
            @variable(model, w[1:I] >= 0)
            @objective(model, Min, sum(data.costs .* y))
            
            # ERROR: gbc_sense contains strings "U" and "F", should only contain GBCBoundType values
            gbc_lhs = [y[1,1]; y[1,2]]
            gbc_rhs = [x[1] + x[2]; x[2]]
            gbc_sense = ["U"; "F"]
            return gbc_lhs, gbc_rhs, gbc_sense
        end
        
        err = try
            ClassicalOracle(data, master; model = update_sub_sense_error_model!)
            nothing
        catch e
            e
        end

        @test err isa ArgumentError
        @test occursin("is not a GBCBoundType", err.msg)
    end
end
