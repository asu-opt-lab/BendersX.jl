"""
UnifiedOracle Test Suite

Tests the UnifiedOracle implementation for Benders decomposition.
Compares results between ClassicalOracle and UnifiedOracle to ensure consistency.

Uses a simple assignment problem structure for testing.
"""

using BendersX
using Test
using JuMP
using HiGHS

# =============================================================================
# Test Problem: Simple Assignment Problem
# =============================================================================

struct SimpleUnifiedTestData
    n_facilities::Int
    n_customers::Int
    costs::Matrix{Float64}
end

function create_unified_test_data(n_facilities::Int=3, n_customers::Int=4)
    # Use fixed seed for reproducibility
    costs = [1.0 2.0 3.0 4.0;
             5.0 1.0 2.0 3.0;
             4.0 3.0 1.0 2.0]
    return SimpleUnifiedTestData(n_facilities, n_customers, costs)
end

function update_master_unified_model!(model::Model, data::SimpleUnifiedTestData)
    optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
    set_optimizer(model, optimizer)
    
    @variable(model, x[1:data.n_facilities], Bin)
    @variable(model, t >= -1e6)
    @objective(model, Min, t)
    
    # At least one facility must be open
    @constraint(model, sum(x) >= 1)
    
    return (x = x, ), t
end

function solve_mip_unified_reference(data::SimpleUnifiedTestData)
    model = Model()
    optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
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

@testset verbose = true "UnifiedOracle Test Suite" begin
    
    @testset "Basic UnifiedOracle Construction" begin
        # Test that UnifiedOracle can be constructed
        data = create_unified_test_data()
        
        function update_sub_basic_model!(model::Model, data::SimpleUnifiedTestData, scen_idx::Int; x)
            optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
            set_optimizer(model, optimizer)
            
            I, J = data.n_facilities, data.n_customers
            @variable(model, y[1:I, 1:J] >= 0)
            
            @objective(model, Min, sum(data.costs .* y))
            @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
            @constraint(model, facility_open[i in 1:I, j in 1:J], y[i,j] <= x[i])
            
            return nothing
        end
        
        master = Master(data; model = update_master_unified_model!)
        oracle = UnifiedOracle(data, master; model = update_sub_basic_model!)
        
        # Verify oracle has all required fields
        @test !isempty(oracle.fixing_lb_constraints)
        @test !isempty(oracle.fixing_ub_constraints)
        @test oracle.objective_constraint isa ConstraintRef
        @test oracle.model isa Model
    end
    
    @testset "UnifiedOracle vs ClassicalOracle - Basic Problem" begin
        # Compare results between ClassicalOracle and UnifiedOracle
        data = create_unified_test_data()
        mip_opt = solve_mip_unified_reference(data)
        
        function update_sub_compare_model!(model::Model, data::SimpleUnifiedTestData, scen_idx::Int; x)
            optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
            set_optimizer(model, optimizer)
            
            I, J = data.n_facilities, data.n_customers
            @variable(model, y[1:I, 1:J] >= 0)
            
            @objective(model, Min, sum(data.costs .* y))
            @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
            @constraint(model, facility_open[i in 1:I, j in 1:J], y[i,j] <= x[i])
            
            return nothing
        end
        
        benders_param = BendersSeqParam(time_limit = 60.0, gap_tolerance = 1e-6, verbose = false)
        
        # Test ClassicalOracle
        master_classical = Master(data; model = update_master_unified_model!)
        oracle_classical = ClassicalOracle(data, master_classical; model = update_sub_compare_model!)
        env_classical = BendersSeq(master_classical, oracle_classical; param = benders_param)
        solve!(env_classical)
        
        @test env_classical.termination_status == Optimal()
        @test isapprox(mip_opt, env_classical.obj_value, atol=1e-4)
        
        # Test UnifiedOracle
        master_unified = Master(data; model = update_master_unified_model!)
        oracle_unified = UnifiedOracle(data, master_unified; model = update_sub_compare_model!)
        env_unified = BendersSeq(master_unified, oracle_unified; param = benders_param)
        solve!(env_unified)
        
        @test env_unified.termination_status == Optimal()
        @test isapprox(mip_opt, env_unified.obj_value, atol=1e-4)
        
        # Both should find the same solution
        @test isapprox(env_classical.obj_value, env_unified.obj_value, atol=1e-4)
    end
    
    @testset "UnifiedOracle Parameters" begin
        # Test that UnifiedOracleParam can be customized
        param = UnifiedOracleParam(rtol = 1e-8, atol = 1e-6, zero_tol = 1e-5)
        
        @test param.rtol == 1e-8
        @test param.atol == 1e-6
        @test param.zero_tol == 1e-5
        @test param.w0 == 1.0  # Default value
        
        # Test w0 parameter
        param_w0 = UnifiedOracleParam(w0 = 2.5)
        @test param_w0.w0 == 2.5
        
        # Test w0 validation - must be positive
        @test_throws ArgumentError UnifiedOracleParam(w0 = 0.0)
        @test_throws ArgumentError UnifiedOracleParam(w0 = -1.0)
        
        data = create_unified_test_data()
        
        function update_sub_param_model!(model::Model, data::SimpleUnifiedTestData, scen_idx::Int; x)
            optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
            set_optimizer(model, optimizer)
            
            I, J = data.n_facilities, data.n_customers
            @variable(model, y[1:I, 1:J] >= 0)
            
            @objective(model, Min, sum(data.costs .* y))
            @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
            @constraint(model, facility_open[i in 1:I, j in 1:J], y[i,j] <= x[i])
            
            return nothing
        end
        
        master = Master(data; model = update_master_unified_model!)
        oracle = UnifiedOracle(data, master; model = update_sub_param_model!, param = param)
        
        @test oracle.param.rtol == 1e-8
        @test oracle.param.zero_tol == 1e-5
        @test oracle.param.w0 == 1.0
    end
    
    @testset "UnifiedOracle with custom w0" begin
        # Test that UnifiedOracle works correctly with custom w0
        data = create_unified_test_data()
        mip_opt = solve_mip_unified_reference(data)
        
        function update_sub_w0_model!(model::Model, data::SimpleUnifiedTestData, scen_idx::Int; x)
            optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
            set_optimizer(model, optimizer)
            
            I, J = data.n_facilities, data.n_customers
            @variable(model, y[1:I, 1:J] >= 0)
            
            @objective(model, Min, sum(data.costs .* y))
            @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
            @constraint(model, facility_open[i in 1:I, j in 1:J], y[i,j] <= x[i])
            
            return nothing
        end
        
        benders_param = BendersSeqParam(time_limit = 60.0, gap_tolerance = 1e-6, verbose = false)
        
        # Test with w0 = 2.0
        param_w0 = UnifiedOracleParam(w0 = 2.0)
        master = Master(data; model = update_master_unified_model!)
        oracle = UnifiedOracle(data, master; model = update_sub_w0_model!, param = param_w0)
        env = BendersSeq(master, oracle; param = benders_param)
        solve!(env)
        
        @test env.termination_status == Optimal()
        @test isapprox(mip_opt, env.obj_value, atol=1e-4)
    end
    
    @testset "Error Handling" begin
        data = create_unified_test_data()

        # Test: ArgumentError for invalid w0
        @testset "Invalid w0 throws ArgumentError" begin
            @test_throws ArgumentError UnifiedOracleParam(w0 = 0.0)
            @test_throws ArgumentError UnifiedOracleParam(w0 = -1.0)
            @test_throws ArgumentError UnifiedOracleParam(w0 = -100.0)
        end
    end

    @testset "Interval Constraint Accepted" begin
        # Test that interval constraints are properly split and accepted
        data = create_unified_test_data()

        function update_sub_interval_accept_model!(model::Model, data::SimpleUnifiedTestData, scen_idx::Int; x)
            optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
            set_optimizer(model, optimizer)

            I, J = data.n_facilities, data.n_customers
            @variable(model, y[1:I, 1:J] >= 0)

            @objective(model, Min, sum(data.costs .* y))
            @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
            @constraint(model, facility_open[i in 1:I, j in 1:J], y[i,j] <= x[i])

            # Interval constraint — should be accepted
            @constraint(model, interval_con, 0.5 <= sum(y) <= 10.0)

            return nothing
        end

        master = Master(data; model = update_master_unified_model!)
        # Should NOT throw
        oracle = UnifiedOracle(data, master; model = update_sub_interval_accept_model!)
        @test oracle.model isa Model
    end

    @testset "Interval Constraint Reformulation" begin
        # Verify that interval constraints are split and σ is correctly applied
        struct UnifiedIntervalReformData
            n::Int
        end

        function update_master_interval_reform_model!(model::Model, data::UnifiedIntervalReformData)
            optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
            set_optimizer(model, optimizer)
            @variable(model, x[1:data.n], Bin)
            @variable(model, t >= -1e6)
            @objective(model, Min, t)
            @constraint(model, sum(x) >= 1)
            return (x = x,), t
        end

        data = UnifiedIntervalReformData(2)

        function update_sub_interval_reform_model!(model::Model, data::UnifiedIntervalReformData, scen_idx::Int; x)
            optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
            set_optimizer(model, optimizer)
            @variable(model, y >= 0)
            @objective(model, Min, y)
            # Interval: 2.0 <= y + x[1] <= 8.0  (contains decision var x)
            @constraint(model, iv_con, 2.0 <= y + x[1] <= 8.0)
            return nothing
        end

        master = Master(data; model = update_master_interval_reform_model!)
        oracle = UnifiedOracle(data, master; model = update_sub_interval_reform_model!)

        σ = variable_by_name(oracle.model, "σ")

        # Interval should be split into _lb (>= 2.0) and _ub (<= 8.0)
        # Then unified transform adds +σ to >= and -σ to <=
        lb_con = constraint_by_name(oracle.model, "iv_con_lb")
        ub_con = constraint_by_name(oracle.model, "iv_con_ub")
        @test lb_con !== nothing
        @test ub_con !== nothing
        @test normalized_coefficient(lb_con, σ) == 1.0   # >= gets +σ
        @test normalized_coefficient(ub_con, σ) == -1.0   # <= gets -σ
    end

    @testset "Vectorized Constraints Accepted" begin
        # Test that vectorized constraints (A*y == b form) are properly split and accepted
        data = create_unified_test_data()

        function update_sub_vectorized_model!(model::Model, data::SimpleUnifiedTestData, scen_idx::Int; x)
            optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
            set_optimizer(model, optimizer)

            I, J = data.n_facilities, data.n_customers
            @variable(model, y[1:I, 1:J] >= 0)

            @objective(model, Min, sum(data.costs .* y))
            # Vectorized equality (VectorAffineFunction in Zeros)
            A_demand = zeros(J, I * J)
            for j in 1:J
                for i in 1:I
                    A_demand[j, (i-1)*J+j] = 1.0
                end
            end
            y_vec = vec(y)
            b_demand = ones(J)
            @constraint(model, demand, A_demand * y_vec == b_demand)
            # Broadcasted form
            @constraint(model, facility_open[i in 1:I, j in 1:J], y[i,j] <= x[i])

            return nothing
        end

        master = Master(data; model = update_master_unified_model!)
        # Should NOT throw
        oracle = UnifiedOracle(data, master; model = update_sub_vectorized_model!)
        @test oracle.model isa Model
    end

    @testset "All Allowed Model Patterns" begin
        # Test with ALL valid patterns from test_validate_LP.jl
        data = create_unified_test_data()

        function update_sub_all_patterns_model!(model::Model, data::SimpleUnifiedTestData, scen_idx::Int; x)
            optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
            set_optimizer(model, optimizer)

            I, J = data.n_facilities, data.n_customers
            # Various variable forms
            @variable(model, y[1:I, 1:J] >= 0)
            @variable(model, 0.1 <= z[i=1:I] <= 10.0)  # double-bounded

            @objective(model, Min, sum(data.costs .* y) + sum(z))

            # Standard constraints
            @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
            @constraint(model, facility_open[i in 1:I, j in 1:J], y[i,j] <= x[i])

            # Interval constraint
            @constraint(model, iv, 1.0 <= sum(z) <= 30.0)

            # Broadcasted vectorized form
            @constraint(model, z .<= fill(10.0, I))
            @constraint(model, z .>= fill(0.1, I))

            return nothing
        end

        master = Master(data; model = update_master_unified_model!)
        # Should NOT throw
        oracle = UnifiedOracle(data, master; model = update_sub_all_patterns_model!)
        @test oracle.model isa Model
    end

    @testset "UnifiedOracle Convergence with Interval Constraints" begin
        # Verify Benders still converges correctly when subproblem has interval constraints
        data = create_unified_test_data()
        mip_opt = solve_mip_unified_reference(data)

        function update_sub_interval_converge_model!(model::Model, data::SimpleUnifiedTestData, scen_idx::Int; x)
            optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
            set_optimizer(model, optimizer)

            I, J = data.n_facilities, data.n_customers
            @variable(model, y[1:I, 1:J] >= 0)

            @objective(model, Min, sum(data.costs .* y))
            @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
            @constraint(model, facility_open[i in 1:I, j in 1:J], y[i,j] <= x[i])
            # Interval constraint that does NOT change optimal solution (loose bound)
            @constraint(model, 0.0 <= sum(y) <= 100.0)

            return nothing
        end

        benders_param = BendersSeqParam(time_limit = 60.0, gap_tolerance = 1e-6, verbose = false)

        master = Master(data; model = update_master_unified_model!)
        oracle = UnifiedOracle(data, master; model = update_sub_interval_converge_model!)
        env = BendersSeq(master, oracle; param = benders_param)
        solve!(env)

        @test env.termination_status == Optimal()
        @test isapprox(mip_opt, env.obj_value, atol = 1e-4)
    end

    @testset "Reformulation Verification" begin
        # Test that the unified reformulation produces the correct model structure

        # Simple test problem with known structure
        struct ReformTestData
            n::Int  # number of decision variables
        end

        function update_master_reform_model!(model::Model, data::ReformTestData)
            optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
            set_optimizer(model, optimizer)

            @variable(model, x[1:data.n], Bin)
            @variable(model, t >= -1e6)
            @objective(model, Min, t)
            @constraint(model, sum(x) >= 1)

            return (x = x, ), t
        end

        @testset "σ variable structure" begin
            # Test that σ variable is free (no lower/upper bound)
            data = ReformTestData(2)

            function update_sub_sigma_model!(model::Model, data::ReformTestData, scen_idx::Int; x)
                optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
                set_optimizer(model, optimizer)

                @variable(model, y >= 0)
                @objective(model, Min, y)
                @constraint(model, con1, y >= x[1] + x[2])

                return nothing
            end

            master = Master(data; model = update_master_reform_model!)
            oracle = UnifiedOracle(data, master; model = update_sub_sigma_model!)

            # Find σ variable by name
            σ = variable_by_name(oracle.model, "σ")
            @test σ !== nothing
            @test !has_lower_bound(σ)
            @test !has_upper_bound(σ)
        end

        @testset "Objective becomes min σ" begin
            data = ReformTestData(2)

            function update_sub_obj_model!(model::Model, data::ReformTestData, scen_idx::Int; x)
                optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
                set_optimizer(model, optimizer)

                @variable(model, y >= 0)
                @objective(model, Min, 3.0 * y + 5.0)  # Original objective
                @constraint(model, con1, y >= x[1] + x[2])

                return nothing
            end

            master = Master(data; model = update_master_reform_model!)
            oracle = UnifiedOracle(data, master; model = update_sub_obj_model!)

            # New objective should be just σ
            σ = variable_by_name(oracle.model, "σ")
            obj = objective_function(oracle.model)
            @test obj == σ
            @test objective_sense(oracle.model) == MOI.MIN_SENSE
        end

        @testset "Fixing constraints structure" begin
            # Verify lb (x + σ >= value) and ub (x - σ <= value) constraints
            data = ReformTestData(2)

            function update_sub_fix_model!(model::Model, data::ReformTestData, scen_idx::Int; x)
                optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
                set_optimizer(model, optimizer)

                @variable(model, y >= 0)
                @objective(model, Min, y)
                @constraint(model, con1, y >= x[1] + x[2])

                return nothing
            end

            master = Master(data; model = update_master_reform_model!)
            oracle = UnifiedOracle(data, master; model = update_sub_fix_model!)

            # Should have one lb and one ub constraint per decision variable
            @test length(oracle.fixing_lb_constraints) == data.n
            @test length(oracle.fixing_ub_constraints) == data.n

            σ = variable_by_name(oracle.model, "σ")

            # Verify lb constraints: x + σ >= 0 (initial RHS)
            for con in oracle.fixing_lb_constraints
                con_obj = constraint_object(con)
                @test con_obj.set isa MOI.GreaterThan
                # σ coefficient should be +1
                @test normalized_coefficient(con, σ) == 1.0
            end

            # Verify ub constraints: x - σ <= 0 (initial RHS)
            for con in oracle.fixing_ub_constraints
                con_obj = constraint_object(con)
                @test con_obj.set isa MOI.LessThan
                # σ coefficient should be -1
                @test normalized_coefficient(con, σ) == -1.0
            end
        end

        @testset "Objective constraint structure" begin
            # Verify original objective becomes constraint: -obj + w0*σ >= -t
            data = ReformTestData(2)
            w0 = 2.5

            function update_sub_objcon_model!(model::Model, data::ReformTestData, scen_idx::Int; x)
                optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
                set_optimizer(model, optimizer)

                @variable(model, y >= 0)
                @objective(model, Min, 3.0 * y)  # Original objective
                @constraint(model, con1, y >= x[1] + x[2])

                return nothing
            end

            master = Master(data; model = update_master_reform_model!)
            param = UnifiedOracleParam(w0 = w0)
            oracle = UnifiedOracle(data, master; model = update_sub_objcon_model!, param = param)

            σ = variable_by_name(oracle.model, "σ")

            # objective_constraint: -original_obj + w0*σ >= -t
            con_obj = constraint_object(oracle.objective_constraint)
            @test con_obj.set isa MOI.GreaterThan

            # σ coefficient should be w0
            @test normalized_coefficient(oracle.objective_constraint, σ) == w0
        end

        @testset "Problem constraints with σ relaxation" begin
            # Test that problem constraints are correctly modified with ±σ
            data = ReformTestData(2)

            function update_sub_relax_model!(model::Model, data::ReformTestData, scen_idx::Int; x)
                optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
                set_optimizer(model, optimizer)

                @variable(model, y >= 0)
                @objective(model, Min, y)
                # >= constraint should get +σ
                @constraint(model, geq_con, y + x[1] >= 5.0)
                # <= constraint should get -σ
                @constraint(model, leq_con, y + x[2] <= 10.0)

                return nothing
            end

            master = Master(data; model = update_master_reform_model!)
            oracle = UnifiedOracle(data, master; model = update_sub_relax_model!)

            σ = variable_by_name(oracle.model, "σ")

            # Find >= constraint and verify σ coefficient is +1
            geq_con = constraint_by_name(oracle.model, "geq_con")
            @test geq_con !== nothing
            @test normalized_coefficient(geq_con, σ) == 1.0

            # Find <= constraint and verify σ coefficient is -1
            leq_con = constraint_by_name(oracle.model, "leq_con")
            @test leq_con !== nothing
            @test normalized_coefficient(leq_con, σ) == -1.0
        end

        @testset "Equality constraints split into lb/ub" begin
            # Test that == constraints are split into >= and <= with ±σ
            data = ReformTestData(2)

            function update_sub_eq_model!(model::Model, data::ReformTestData, scen_idx::Int; x)
                optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
                set_optimizer(model, optimizer)

                @variable(model, y >= 0)
                @objective(model, Min, y)
                # == constraint should be split
                @constraint(model, eq_con, y + x[1] == 5.0)

                return nothing
            end

            master = Master(data; model = update_master_reform_model!)
            oracle = UnifiedOracle(data, master; model = update_sub_eq_model!)

            σ = variable_by_name(oracle.model, "σ")

            # Original eq_con should be deleted, replaced by eq_con_lb and eq_con_ub
            @test constraint_by_name(oracle.model, "eq_con") === nothing

            # lb version should exist with σ coefficient +1
            eq_con_lb = constraint_by_name(oracle.model, "eq_con_lb")
            @test eq_con_lb !== nothing
            con_obj_lb = constraint_object(eq_con_lb)
            @test con_obj_lb.set isa MOI.GreaterThan
            @test normalized_coefficient(eq_con_lb, σ) == 1.0

            # ub version should exist with σ coefficient -1
            eq_con_ub = constraint_by_name(oracle.model, "eq_con_ub")
            @test eq_con_ub !== nothing
            con_obj_ub = constraint_object(eq_con_ub)
            @test con_obj_ub.set isa MOI.LessThan
            @test normalized_coefficient(eq_con_ub, σ) == -1.0
        end

        @testset "Constraints without decision variables not relaxed" begin
            # Constraints not containing x should NOT have σ added
            data = ReformTestData(2)

            function update_sub_nox_model!(model::Model, data::ReformTestData, scen_idx::Int; x)
                optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
                set_optimizer(model, optimizer)

                @variable(model, y >= 0)
                @variable(model, z >= 0)
                @objective(model, Min, y + z)
                # Constraint with x - should be relaxed
                @constraint(model, with_x, y + x[1] >= 1.0)
                # Constraint without x - should NOT be relaxed
                @constraint(model, without_x, y + z >= 2.0)

                return nothing
            end

            master = Master(data; model = update_master_reform_model!)
            oracle = UnifiedOracle(data, master; model = update_sub_nox_model!)

            σ = variable_by_name(oracle.model, "σ")

            # Constraint with x should have σ
            with_x_con = constraint_by_name(oracle.model, "with_x")
            @test normalized_coefficient(with_x_con, σ) == 1.0

            # Constraint without x should NOT have σ
            without_x_con = constraint_by_name(oracle.model, "without_x")
            @test normalized_coefficient(without_x_con, σ) == 0.0
        end
    end
end
