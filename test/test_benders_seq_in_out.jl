using Test
using JuMP
using GLPK
using BendersX

# -----------------------------------------------------------------------------
# Test data and models
# -----------------------------------------------------------------------------

struct InOutTestData end

function build_master!(model::Model, ::InOutTestData)
    @variable(model, 0 <= x[1:1] <= 1)
    @variable(model, t >= 0)
    @objective(model, Min, x[1] + t)
    return (x = x,), t
end

function build_subproblem!(model::Model, ::InOutTestData, ::Int; x)
    @variable(model, y >= 0)
    @constraint(model, y >= 2 - x[1])
    @objective(model, Min, y)
    return nothing
end

function make_master()
    return Master(
        InOutTestData();
        model = build_master!,
        optimizer = GLPK.Optimizer,
    )
end

function make_oracle(master)
    return ClassicalOracle(
        InOutTestData(),
        master;
        model = build_subproblem!,
        optimizer = GLPK.Optimizer,
    )
end

# -----------------------------------------------------------------------------
# Recording Oracle
# -----------------------------------------------------------------------------

mutable struct RecordingOracle <: BendersX.AbstractOracle
    is_in_L_sequence::Vector{Bool}
    x_values::Vector{Vector{Float64}}
    t_values::Vector{Vector{Float64}}
    objective_value::Float64
    dimension::Int
end

BendersX.auxiliary_dimension(oracle::RecordingOracle) = oracle.dimension

function BendersX.generate_cuts(
    oracle::RecordingOracle,
    x_value::Vector{Float64},
    t_value::Vector{Float64};
    tol_normalize = 1.0,
    time_limit = 3600.0,
)
    push!(oracle.x_values, copy(x_value))
    push!(oracle.t_values, copy(t_value))

    k = min(length(oracle.x_values), length(oracle.is_in_L_sequence))
    is_in_L = oracle.is_in_L_sequence[k]

    cut = BendersX.Hyperplane(length(x_value), oracle.dimension)
    cut.a_t[1] = -1.0
    cut.a_0 = oracle.objective_value

    return is_in_L, [cut], [oracle.objective_value]
end

function recording_oracle(sequence; dimension = 1, objective_value = 1.0)
    return RecordingOracle(
        collect(sequence),
        Vector{Vector{Float64}}(),
        Vector{Vector{Float64}}(),
        objective_value,
        dimension,
    )
end

# -----------------------------------------------------------------------------
# Parameter tests
# -----------------------------------------------------------------------------

@testset "BendersSeqInOut" begin
    @testset "parameter validation" begin
        @test BendersSeqInOutParam(
            stabilizing_x = [0.0],
            α = 0.0,
            λ = 0.0,
        ) isa BendersSeqInOutParam

        @test BendersSeqInOutParam(
            stabilizing_x = [0.0],
            α = 1.0,
            λ = 1.0,
        ) isa BendersSeqInOutParam

        @test_throws ArgumentError BendersSeqInOutParam(
            stabilizing_x = [0.0],
            α = -0.1,
        )
        @test_throws ArgumentError BendersSeqInOutParam(
            stabilizing_x = [0.0],
            α = 1.1,
        )
        @test_throws ArgumentError BendersSeqInOutParam(
            stabilizing_x = [0.0],
            λ = -0.1,
        )
        @test_throws ArgumentError BendersSeqInOutParam(
            stabilizing_x = [0.0],
            λ = 1.1,
        )
    end

    # -------------------------------------------------------------------------
    # Basic end-to-end test
    # -------------------------------------------------------------------------

    @testset "solves a small problem" begin
        master = make_master()
        oracle = make_oracle(master)
        param = BendersSeqInOutParam(
            stabilizing_x = [1.0],
            α = 1.0,
            λ = 0.5,
            time_limit = 30.0,
            gap_tolerance = 1.0e-8,
            verbose = false,
        )

        env = BendersSeqInOut(master, oracle; param = param)
        result = solve!(env)

        @test env.termination_status == Optimal()
        @test isapprox(env.obj_value, 2.0; atol = 1.0e-8)
        @test !isempty(result)
    end

    # -------------------------------------------------------------------------
    # In-out query behavior
    # -------------------------------------------------------------------------

    @testset "queries a perturbed point" begin
        master = make_master()
        oracle = recording_oracle([false]; objective_value = 2.0)
        param = BendersSeqInOutParam(
            stabilizing_x = [1.0],
            α = 1.0,
            λ = 0.5,
            time_limit = 30.0,
            gap_tolerance = 1.0e-8,
            verbose = false,
        )

        env = BendersSeqInOut(master, oracle; param = param)
        solve!(env)

        @test length(oracle.x_values) >= 1
        # The first master candidate is x = 0. With α = 1, the stabilizing
        # point remains 1, so the first query is the midpoint.
        @test isapprox(oracle.x_values[1][1], 0.5; atol = 1.0e-12)
    end

    # -------------------------------------------------------------------------
    # Intermediate point moves toward the candidate when it is in L
    # -------------------------------------------------------------------------

    @testset "moves the query point toward the candidate when in L" begin
        master = make_master()
        oracle = recording_oracle([true, true, false]; objective_value = 2.0)
        param = BendersSeqInOutParam(
            stabilizing_x = [1.0],
            α = 1.0,
            λ = 0.1,
            time_limit = 30.0,
            gap_tolerance = 1.0e-8,
            verbose = false,
        )

        env = BendersSeqInOut(master, oracle; param = param)
        solve!(env)

        @test length(oracle.x_values) >= 3
        query_x = [x[1] for x in oracle.x_values]
        @test all(0.0 <= x <= 1.0 for x in query_x)
        @test query_x[1] > query_x[2] > query_x[3]
    end

    # -------------------------------------------------------------------------
    # Kelley switch when λ reaches 1
    # -------------------------------------------------------------------------

    @testset "switches to Kelley mode when λ reaches 1" begin
        master = make_master()
        oracle = recording_oracle(fill(true, 20); objective_value = 2.0)
        param = BendersSeqInOutParam(
            stabilizing_x = [1.0],
            α = 1.0,
            λ = 0.8,
            time_limit = 30.0,
            gap_tolerance = 1.0e-8,
            halt_limit = 1,
            verbose = true,
        )

        env = BendersSeqInOut(master, oracle; param = param)
        solve!(env)

        # Starting from λ = 0.8, the initial query is 0.8 * 0 + 0.2 * 1 = 0.2.
        # The in-out loop then moves through λ = 0.9 and λ = 1.0 before
        # entering Kelley mode.
        @test length(oracle.x_values) >= 3
        @test isapprox(oracle.x_values[1][1], 0.2; atol = 1.0e-12)
        @test isapprox(oracle.x_values[2][1], 0.1; atol = 1.0e-12)
        @test isapprox(oracle.x_values[3][1], 0.0; atol = 1.0e-12)
    end
end
