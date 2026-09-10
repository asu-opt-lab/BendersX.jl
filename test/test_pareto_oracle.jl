"""
ParetoOracle Test Suite

Tests the ParetoOracle implementation for Benders decomposition.
Compares results between ClassicalOracle and ParetoOracle to ensure consistency.

Uses a simple assignment problem structure for testing.
"""

using BendersX
import BendersX: generate_cuts, Hyperplane
using Test
using JuMP
using HiGHS

# =============================================================================
# Test Problem: Simple Assignment Problem
# =============================================================================

struct SimpleParetoTestData
    n_facilities::Int
    n_customers::Int
    costs::Matrix{Float64}
end

function create_pareto_test_data(n_facilities::Int=3, n_customers::Int=4)
    # Use fixed seed for reproducibility
    costs = [1.0 2.0 3.0 4.0;
             5.0 1.0 2.0 3.0;
             4.0 3.0 1.0 2.0]
    return SimpleParetoTestData(n_facilities, n_customers, costs)
end

function update_master_pareto_model!(model::Model, data::SimpleParetoTestData)
    optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
    set_optimizer(model, optimizer)
    
    @variable(model, x[1:data.n_facilities], Bin)
    @variable(model, t >= -1e6)
    @objective(model, Min, t)
    
    # At least one facility must be open
    @constraint(model, sum(x) >= 1)
    
    return (x = x, ), t
end

function solve_mip_pareto_reference(data::SimpleParetoTestData)
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

function update_sub_pareto_model!(model::Model, data::SimpleParetoTestData, scen_idx::Int; x)
    optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
    set_optimizer(model, optimizer)
    
    I, J = data.n_facilities, data.n_customers
    @variable(model, y[1:I, 1:J] >= 0)
    
    @objective(model, Min, sum(data.costs .* y))
    @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
    @constraint(model, facility_open[i in 1:I, j in 1:J], y[i,j] <= x[i])
    
    return nothing
end

# =============================================================================
# Test Scenarios
# =============================================================================

@testset verbose = true "ParetoOracle Test Suite" begin
    
    @testset "ParetoOracleParam Construction" begin
        # Test that core_point is required
        @test_throws MethodError ParetoOracleParam()
        
        # Test that empty core_point throws error
        @test_throws ArgumentError ParetoOracleParam(Float64[])
        
        # Test valid construction
        param = ParetoOracleParam([0.5, 0.5, 0.5])
        @test param.core_point == [0.5, 0.5, 0.5]
        @test param.rtol == 1e-9
        @test param.atol == 0.0
        @test param.zero_tol == 1e-9
        @test param.λ == 1.0  # Default λ

        # Test with custom tolerances
        param2 = ParetoOracleParam([0.3, 0.7]; rtol = 1e-8, atol = 1e-6, zero_tol = 1e-5)
        @test param2.core_point == [0.3, 0.7]
        @test param2.rtol == 1e-8
        @test param2.atol == 1e-6
        @test param2.zero_tol == 1e-5
        @test param2.λ == 1.0  # Default λ
        
        # Test with custom λ
        param3 = ParetoOracleParam([0.5, 0.5]; λ = 0.8)
        @test param3.λ == 0.8
        
        # Test λ must be in [0, 1]
        @test_throws ArgumentError ParetoOracleParam([0.5]; λ = -0.1)
        @test_throws ArgumentError ParetoOracleParam([0.5]; λ = 1.5)
    end
    
    @testset "Basic ParetoOracle Construction" begin
        data = create_pareto_test_data()
        master = Master(data; model = update_master_pareto_model!)
        
        # core_point dimension must match decision variables
        param = ParetoOracleParam(fill(0.5, data.n_facilities))
        oracle = ParetoOracle(data, master, param; model = update_sub_pareto_model!)
        
        # Verify oracle has all required fields
        @test !isempty(oracle.fixed_x_constraints)
        @test !isempty(oracle.pareto_model[:pareto_fixing_constraints])
        @test oracle.pareto_model[:σ] isa VariableRef
        @test oracle.model isa Model
        @test oracle.pareto_model isa Model
        @test oracle.param.core_point == fill(0.5, data.n_facilities)
    end
    
    @testset "ParetoOracle Dimension Mismatch" begin
        data = create_pareto_test_data()
        master = Master(data; model = update_master_pareto_model!)
        
        # Wrong dimension for core_point should throw
        param_wrong = ParetoOracleParam([0.5, 0.5])  # Only 2 elements, but n_facilities = 3
        @test_throws DimensionMismatch ParetoOracle(data, master, param_wrong; model = update_sub_pareto_model!)
    end
    
    @testset "Interval Constraint Accepted" begin
        # Test that interval constraints are properly split and accepted
        data = create_pareto_test_data()
        master = Master(data; model = update_master_pareto_model!)
        param = ParetoOracleParam(fill(0.5, data.n_facilities))

        function update_sub_with_interval_model!(model::Model, data::SimpleParetoTestData, scen_idx::Int; x)
            optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
            set_optimizer(model, optimizer)

            I, J = data.n_facilities, data.n_customers
            @variable(model, y[1:I, 1:J] >= 0)

            @objective(model, Min, sum(data.costs .* y))
            @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
            @constraint(model, facility_open[i in 1:I, j in 1:J], y[i,j] <= x[i])

            # Add interval constraint — should be accepted, not rejected
            @constraint(model, interval_con, 0.5 <= sum(y) <= 10.0)

            return nothing
        end

        # Should NOT throw
        oracle = ParetoOracle(data, master, param; model = update_sub_with_interval_model!)
        @test oracle.model isa Model
        @test oracle.pareto_model isa Model
    end

    @testset "Interval Constraint Reformulation" begin
        # Verify σ coefficients are correct after splitting interval constraints
        struct ParetoIntervalReformData
            n::Int
        end

        function update_master_interval_reform_model!(model::Model, data::ParetoIntervalReformData)
            optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
            set_optimizer(model, optimizer)
            @variable(model, x[1:data.n], Bin)
            @variable(model, t >= -1e6)
            @objective(model, Min, t)
            @constraint(model, sum(x) >= 1)
            return (x = x,), t
        end

        data = ParetoIntervalReformData(2)

        function update_sub_interval_reform_model!(model::Model, data::ParetoIntervalReformData, scen_idx::Int; x)
            optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
            set_optimizer(model, optimizer)
            @variable(model, y >= 0)
            @objective(model, Min, y)
            # Interval constraint: 2.0 <= y + x[1] <= 8.0
            @constraint(model, iv_con, 2.0 <= y + x[1] <= 8.0)
            return nothing
        end

        master = Master(data; model = update_master_interval_reform_model!)
        param = ParetoOracleParam(fill(0.5, data.n))
        oracle = ParetoOracle(data, master, param; model = update_sub_interval_reform_model!)

        σ = oracle.pareto_model[:σ]

        # Interval should be split into _lb (>= 2.0) and _ub (<= 8.0)
        lb_con = constraint_by_name(oracle.pareto_model, "iv_con_lb")
        ub_con = constraint_by_name(oracle.pareto_model, "iv_con_ub")
        @test lb_con !== nothing
        @test ub_con !== nothing
        # σ coefficient should equal RHS for each split constraint
        @test normalized_coefficient(lb_con, σ) == 2.0
        @test normalized_coefficient(ub_con, σ) == 8.0
    end

    @testset "Vectorized Constraints Accepted" begin
        # Test that vectorized constraints (A*y == b form) are properly split and accepted
        data = create_pareto_test_data()
        master = Master(data; model = update_master_pareto_model!)
        param = ParetoOracleParam(fill(0.5, data.n_facilities))

        function update_sub_vectorized_model!(model::Model, data::SimpleParetoTestData, scen_idx::Int; x)
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

        # Should NOT throw
        oracle = ParetoOracle(data, master, param; model = update_sub_vectorized_model!)
        @test oracle.model isa Model
    end

    @testset "All Allowed Model Patterns" begin
        # Test with ALL valid patterns from test_validate_LP.jl
        data = create_pareto_test_data()
        master = Master(data; model = update_master_pareto_model!)
        param = ParetoOracleParam(fill(0.5, data.n_facilities))

        function update_sub_all_patterns_model!(model::Model, data::SimpleParetoTestData, scen_idx::Int; x)
            optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
            set_optimizer(model, optimizer)

            I, J = data.n_facilities, data.n_customers
            # Various variable container forms
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

        # Should NOT throw
        oracle = ParetoOracle(data, master, param; model = update_sub_all_patterns_model!)
        @test oracle.model isa Model
        @test oracle.pareto_model isa Model
    end

    @testset "ParetoOracle Convergence with Interval Constraints" begin
        # Verify Benders still converges correctly when subproblem has interval constraints
        data = create_pareto_test_data()
        mip_opt = solve_mip_pareto_reference(data)

        function update_sub_interval_converge_model!(model::Model, data::SimpleParetoTestData, scen_idx::Int; x)
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

        master = Master(data; model = update_master_pareto_model!)
        param = ParetoOracleParam(fill(0.5, data.n_facilities))
        oracle = ParetoOracle(data, master, param; model = update_sub_interval_converge_model!)
        env = BendersSeq(master, oracle; param = benders_param)
        solve!(env)

        @test env.termination_status == Optimal()
        @test isapprox(mip_opt, env.obj_value, atol = 1e-4)
    end
    
    @testset "ParetoOracle vs ClassicalOracle - Basic Problem" begin
        # Compare results between ClassicalOracle and ParetoOracle
        data = create_pareto_test_data()
        mip_opt = solve_mip_pareto_reference(data)
        
        benders_param = BendersSeqParam(time_limit = 60.0, gap_tolerance = 1e-6, verbose = false)
        
        # Test ClassicalOracle
        master_classical = Master(data; model = update_master_pareto_model!)
        oracle_classical = ClassicalOracle(data, master_classical; model = update_sub_pareto_model!)
        env_classical = BendersSeq(master_classical, oracle_classical; param = benders_param)
        solve!(env_classical)
        
        @test env_classical.termination_status == Optimal()
        @test isapprox(mip_opt, env_classical.obj_value, atol=1e-4)
        
        # Test ParetoOracle
        master_pareto = Master(data; model = update_master_pareto_model!)
        pareto_param = ParetoOracleParam(fill(0.5, data.n_facilities))
        oracle_pareto = ParetoOracle(data, master_pareto, pareto_param; model = update_sub_pareto_model!)
        env_pareto = BendersSeq(master_pareto, oracle_pareto; param = benders_param)
        solve!(env_pareto)
        
        @test env_pareto.termination_status == Optimal()
        @test isapprox(mip_opt, env_pareto.obj_value, atol=1e-4)
        
        # Both should find the same solution
        @test isapprox(env_classical.obj_value, env_pareto.obj_value, atol=1e-4)
    end
    
    @testset "ParetoOracle with Different Core Points" begin
        # Note: Core point must be in relative interior of feasible region for x
        # For this problem, x ∈ [0,1]^3 with sum(x) >= 1
        # So core_point = [0.5, 0.5, 0.5] is a valid interior point (sum = 1.5 > 1)
        
        data = create_pareto_test_data()
        mip_opt = solve_mip_pareto_reference(data)
        
        benders_param = BendersSeqParam(time_limit = 60.0, gap_tolerance = 1e-6, verbose = false)
        
        # Test with core_point = [0.5, 0.5, 0.5] (interior point)
        master1 = Master(data; model = update_master_pareto_model!)
        param1 = ParetoOracleParam(fill(0.5, data.n_facilities))
        oracle1 = ParetoOracle(data, master1, param1; model = update_sub_pareto_model!)
        env1 = BendersSeq(master1, oracle1; param = benders_param)
        solve!(env1)
        
        @test env1.termination_status == Optimal()
        @test isapprox(mip_opt, env1.obj_value, atol=1e-4)
        
        # Test with core_point = [0.8, 0.8, 0.8] (another interior point)
        master2 = Master(data; model = update_master_pareto_model!)
        param2 = ParetoOracleParam(fill(0.8, data.n_facilities))
        oracle2 = ParetoOracle(data, master2, param2; model = update_sub_pareto_model!)
        env2 = BendersSeq(master2, oracle2; param = benders_param)
        solve!(env2)
        
        @test env2.termination_status == Optimal()
        @test isapprox(mip_opt, env2.obj_value, atol=1e-4)
        
        # Test with asymmetric interior core_point (sum = 1.5 > 1)
        master3 = Master(data; model = update_master_pareto_model!)
        param3 = ParetoOracleParam([0.4, 0.5, 0.6])
        oracle3 = ParetoOracle(data, master3, param3; model = update_sub_pareto_model!)
        env3 = BendersSeq(master3, oracle3; param = benders_param)
        solve!(env3)
        
        @test env3.termination_status == Optimal()
        @test isapprox(mip_opt, env3.obj_value, atol=1e-4)
    end
    
    @testset "ParetoOracle generate_cuts Basic" begin
        data = create_pareto_test_data()
        master = Master(data; model = update_master_pareto_model!)
        param = ParetoOracleParam(fill(0.5, data.n_facilities))
        oracle = ParetoOracle(data, master, param; model = update_sub_pareto_model!)
        
        # Test generate_cuts with a feasible point
        x_value = [1.0, 0.0, 0.0]  # Only facility 1 open
        t_value = [0.0]  # Initial t value
        
        is_in_L, hyperplanes, sub_obj_vals = generate_cuts(oracle, x_value, t_value)
        
        # Should return valid hyperplane
        @test length(hyperplanes) == 1
        @test length(sub_obj_vals) == 1
        @test sub_obj_vals[1] >= 0  # Objective should be non-negative
        
        # Hyperplane should have correct dimensions
        @test length(hyperplanes[1].a_x) == data.n_facilities
        @test length(hyperplanes[1].a_t) == 1
    end
    
    @testset "ParetoOracle Dynamic Core Point Update (λ parameter)" begin
        # Test that λ < 1 updates core_point after each generate_cuts call
        data = create_pareto_test_data()
        master = Master(data; model = update_master_pareto_model!)
        
        # Test with λ = 0.5 (core_point should move towards x_value)
        initial_core = fill(0.5, data.n_facilities)
        param = ParetoOracleParam(copy(initial_core); λ = 0.5)
        oracle = ParetoOracle(data, master, param; model = update_sub_pareto_model!)
        
        # Verify initial core_point
        @test oracle.param.core_point == initial_core
        
        # First call to generate_cuts
        x_value_1 = [1.0, 0.0, 0.0]
        t_value = [0.0]
        generate_cuts(oracle, x_value_1, t_value)
        
        # core_point should now be: 0.5 * [0.5,0.5,0.5] + 0.5 * [1.0,0.0,0.0] = [0.75, 0.25, 0.25]
        expected_core_1 = 0.5 .* initial_core .+ 0.5 .* x_value_1
        @test isapprox(oracle.param.core_point, expected_core_1, atol=1e-10)
        
        # Second call with different x_value
        x_value_2 = [0.0, 1.0, 0.0]
        generate_cuts(oracle, x_value_2, t_value)
        
        # core_point should now be: 0.5 * [0.75,0.25,0.25] + 0.5 * [0.0,1.0,0.0] = [0.375, 0.625, 0.125]
        expected_core_2 = 0.5 .* expected_core_1 .+ 0.5 .* x_value_2
        @test isapprox(oracle.param.core_point, expected_core_2, atol=1e-10)
    end
    
    @testset "ParetoOracle with λ = 1.0 (no update)" begin
        # Test that λ = 1.0 keeps core_point unchanged (classical behavior)
        data = create_pareto_test_data()
        master = Master(data; model = update_master_pareto_model!)
        
        initial_core = fill(0.5, data.n_facilities)
        param = ParetoOracleParam(copy(initial_core); λ = 1.0)  # Default: no update
        oracle = ParetoOracle(data, master, param; model = update_sub_pareto_model!)
        
        # Verify initial core_point
        @test oracle.param.core_point == initial_core
        
        # Call generate_cuts multiple times
        x_value = [1.0, 0.0, 0.0]
        t_value = [0.0]
        generate_cuts(oracle, x_value, t_value)
        generate_cuts(oracle, x_value, t_value)
        
        # core_point should remain unchanged with λ = 1.0
        @test oracle.param.core_point == initial_core
    end
    
    @testset "ParetoOracle with λ = 0.0 (full update)" begin
        # Test that λ = 0.0 replaces core_point with x_value entirely
        data = create_pareto_test_data()
        master = Master(data; model = update_master_pareto_model!)
        
        initial_core = fill(0.5, data.n_facilities)
        param = ParetoOracleParam(copy(initial_core); λ = 0.0)  # Full update
        oracle = ParetoOracle(data, master, param; model = update_sub_pareto_model!)
        
        # Call generate_cuts
        x_value = [1.0, 0.0, 0.0]
        t_value = [0.0]
        generate_cuts(oracle, x_value, t_value)
        
        # core_point should be exactly x_value
        @test oracle.param.core_point == x_value
    end
    
    @testset "ParetoOracle with Dynamic Core Point - Solution Quality" begin
        # Verify that dynamic core point still produces correct solutions
        data = create_pareto_test_data()
        mip_opt = solve_mip_pareto_reference(data)

        benders_param = BendersSeqParam(time_limit = 60.0, gap_tolerance = 1e-6, verbose = false)

        # Test with λ = 0.8 (gradual update)
        master = Master(data; model = update_master_pareto_model!)
        param = ParetoOracleParam(fill(0.5, data.n_facilities); λ = 0.8)
        oracle = ParetoOracle(data, master, param; model = update_sub_pareto_model!)
        env = BendersSeq(master, oracle; param = benders_param)
        solve!(env)

        @test env.termination_status == Optimal()
        @test isapprox(mip_opt, env.obj_value, atol=1e-4)
    end

    @testset "Reformulation Verification" begin
        # Test that the pareto reformulation produces the correct model structure

        # Simple test problem with known structure
        struct ParetoReformTestData
            n::Int  # number of decision variables
        end

        function update_master_pareto_reform_model!(model::Model, data::ParetoReformTestData)
            optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
            set_optimizer(model, optimizer)

            @variable(model, x[1:data.n], Bin)
            @variable(model, t >= -1e6)
            @objective(model, Min, t)
            @constraint(model, sum(x) >= 1)

            return (x = x, ), t
        end

        @testset "σ variable structure in pareto_model" begin
            # Test that σ variable is added with correct bounds (σ <= 0)
            data = ParetoReformTestData(2)

            function update_sub_sigma_model!(model::Model, data::ParetoReformTestData, scen_idx::Int; x)
                optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
                set_optimizer(model, optimizer)

                @variable(model, y >= 0)
                @objective(model, Min, y)
                @constraint(model, con1, y >= x[1] + x[2])

                return nothing
            end

            master = Master(data; model = update_master_pareto_reform_model!)
            param = ParetoOracleParam(fill(0.5, data.n))
            oracle = ParetoOracle(data, master, param; model = update_sub_sigma_model!)

            # σ is stored as oracle.pareto_model[:σ]
            σ = oracle.pareto_model[:σ]
            @test σ !== nothing
            @test has_upper_bound(σ)
            @test upper_bound(σ) == 0.0
        end

        @testset "pareto_fixing_constraints structure" begin
            # Verify fixing constraints: x + x*·σ = x_0 exist for each decision variable
            data = ParetoReformTestData(2)

            function update_sub_fix_model!(model::Model, data::ParetoReformTestData, scen_idx::Int; x)
                optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
                set_optimizer(model, optimizer)

                @variable(model, y >= 0)
                @objective(model, Min, y)
                @constraint(model, con1, y >= x[1] + x[2])

                return nothing
            end

            master = Master(data; model = update_master_pareto_reform_model!)
            param = ParetoOracleParam(fill(0.5, data.n))
            oracle = ParetoOracle(data, master, param; model = update_sub_fix_model!)

            # Should have one fixing constraint per decision variable
            @test length(oracle.pareto_model[:pareto_fixing_constraints]) == data.n

            σ = oracle.pareto_model[:σ]

            # Verify fixing constraints are equality constraints
            for con in oracle.pareto_model[:pareto_fixing_constraints]
                con_obj = constraint_object(con)
                @test con_obj.set isa MOI.EqualTo
                # Initial σ coefficient is 0 (set in generate_cuts to x*)
                @test normalized_coefficient(con, σ) == 0.0
            end
        end

        @testset "b*σ term added to problem constraints" begin
            # Test that problem constraints get b*σ added (where b is RHS)
            data = ParetoReformTestData(2)

            function update_sub_bsigma_model!(model::Model, data::ParetoReformTestData, scen_idx::Int; x)
                optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
                set_optimizer(model, optimizer)

                @variable(model, y >= 0)
                @objective(model, Min, y)
                # Constraint: y + x[1] >= 5.0, after transformation: y + x[1] + 5*σ >= 5.0
                @constraint(model, geq_con, y + x[1] >= 5.0)
                # Constraint: y + x[2] <= 10.0, after transformation: y + x[2] + 10*σ <= 10.0
                @constraint(model, leq_con, y + x[2] <= 10.0)

                return nothing
            end

            master = Master(data; model = update_master_pareto_reform_model!)
            param = ParetoOracleParam(fill(0.5, data.n))
            oracle = ParetoOracle(data, master, param; model = update_sub_bsigma_model!)

            σ = oracle.pareto_model[:σ]

            # Find >= constraint and verify σ coefficient equals RHS
            geq_con = constraint_by_name(oracle.pareto_model, "geq_con")
            @test geq_con !== nothing
            @test normalized_coefficient(geq_con, σ) == 5.0  # b = 5.0

            # Find <= constraint and verify σ coefficient equals RHS
            leq_con = constraint_by_name(oracle.pareto_model, "leq_con")
            @test leq_con !== nothing
            @test normalized_coefficient(leq_con, σ) == 10.0  # b = 10.0
        end

        @testset "σ added to objective" begin
            # Test that σ is added to objective (initially with 0 coefficient)
            data = ParetoReformTestData(2)

            function update_sub_objsigma_model!(model::Model, data::ParetoReformTestData, scen_idx::Int; x)
                optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
                set_optimizer(model, optimizer)

                @variable(model, y >= 0)
                @objective(model, Min, 3.0 * y + 2.0)  # Original: 3y + 2
                @constraint(model, con1, y >= x[1] + x[2])

                return nothing
            end

            master = Master(data; model = update_master_pareto_reform_model!)
            param = ParetoOracleParam(fill(0.5, data.n))
            oracle = ParetoOracle(data, master, param; model = update_sub_objsigma_model!)

            σ = oracle.pareto_model[:σ]
            obj = objective_function(oracle.pareto_model)

            # Objective should include σ (initially with coefficient 0, set to ξ* during generate_cuts)
            # The coefficient should be 0 initially
            @test coefficient(obj, σ) == 0.0

            # Original objective terms should still be present
            y = variable_by_name(oracle.pareto_model, "y")
            @test coefficient(obj, y) == 3.0

            @test objective_sense(oracle.pareto_model) == MOI.MIN_SENSE
        end

        @testset "Standard model has fixing constraints" begin
            # Verify standard model has fix_x constraints
            data = ParetoReformTestData(2)

            function update_sub_stdfix_model!(model::Model, data::ParetoReformTestData, scen_idx::Int; x)
                optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
                set_optimizer(model, optimizer)

                @variable(model, y >= 0)
                @objective(model, Min, y)
                @constraint(model, con1, y >= x[1] + x[2])

                return nothing
            end

            master = Master(data; model = update_master_pareto_reform_model!)
            param = ParetoOracleParam(fill(0.5, data.n))
            oracle = ParetoOracle(data, master, param; model = update_sub_stdfix_model!)

            # Standard model should have fixing constraints
            @test length(oracle.fixed_x_constraints) == data.n

            # Verify they are equality constraints
            for con in oracle.fixed_x_constraints
                con_obj = constraint_object(con)
                @test con_obj.set isa MOI.EqualTo
            end
        end

        @testset "Equality constraints get b*σ" begin
            # Test that == constraints also get b*σ term
            data = ParetoReformTestData(2)

            function update_sub_eq_model!(model::Model, data::ParetoReformTestData, scen_idx::Int; x)
                optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
                set_optimizer(model, optimizer)

                @variable(model, y >= 0)
                @objective(model, Min, y)
                @constraint(model, eq_con, y + x[1] == 7.0)

                return nothing
            end

            master = Master(data; model = update_master_pareto_reform_model!)
            param = ParetoOracleParam(fill(0.5, data.n))
            oracle = ParetoOracle(data, master, param; model = update_sub_eq_model!)

            σ = oracle.pareto_model[:σ]

            # Find == constraint and verify σ coefficient equals RHS
            eq_con = constraint_by_name(oracle.pareto_model, "eq_con")
            @test eq_con !== nothing
            @test normalized_coefficient(eq_con, σ) == 7.0
        end

        @testset "Zero RHS constraint" begin
            # Test constraint with RHS = 0 gets σ coefficient = 0
            data = ParetoReformTestData(2)

            function update_sub_zero_model!(model::Model, data::ParetoReformTestData, scen_idx::Int; x)
                optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
                set_optimizer(model, optimizer)

                @variable(model, y >= 0)
                @objective(model, Min, y)
                @constraint(model, zero_con, y + x[1] >= 0.0)

                return nothing
            end

            master = Master(data; model = update_master_pareto_reform_model!)
            param = ParetoOracleParam(fill(0.5, data.n))
            oracle = ParetoOracle(data, master, param; model = update_sub_zero_model!)

            σ = oracle.pareto_model[:σ]

            # Constraint with RHS = 0 should have σ coefficient = 0
            zero_con = constraint_by_name(oracle.pareto_model, "zero_con")
            @test zero_con !== nothing
            @test normalized_coefficient(zero_con, σ) == 0.0
        end
    end
end
