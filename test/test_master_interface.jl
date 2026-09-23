using Test
using JuMP
using GLPK
using LinearAlgebra

const MASTER_INTERFACE_CPLEX_AVAILABLE = !isnothing(Base.find_package("CPLEX"))
MASTER_INTERFACE_CPLEX_AVAILABLE && @eval using CPLEX

struct MasterInterfaceTestData end

function build_master_interface_test_model!(model::Model, ::MasterInterfaceTestData)
    @variable(model, 0 <= open[1:1] <= 2)
    @variable(model, theta >= 0)
    @objective(model, Min, open[1] + theta)
    return (open = open,), theta
end

function build_master_interface_test_subproblem!(
    model::Model,
    ::MasterInterfaceTestData,
    ::Int;
    open,
)
    @variable(model, y >= 0)
    @constraint(model, y >= 2 - open[1])
    @objective(model, Min, y)
    return nothing
end

# This test master intentionally uses none of Master's field names. It verifies
# that built-in components depend on the public interface rather than an
# undocumented concrete layout.
mutable struct RenamedFieldMaster <: BendersX.AbstractMaster
    jump_model::Model
    linking_structure::NamedTuple
    linking_vars::Vector{VariableRef}
    auxiliary_vars::Vector{VariableRef}
    linking_costs::Vector{Float64}
    auxiliary_costs::Vector{Float64}
    cut_batches::Int
end

BendersX.master_model(master::RenamedFieldMaster) = master.jump_model
BendersX.linking_variables(master::RenamedFieldMaster) = master.linking_vars
BendersX.auxiliary_variables(master::RenamedFieldMaster) = master.auxiliary_vars
BendersX.copy_linking_variable_tuple!(model::Model, master::RenamedFieldMaster) =
    BendersX.copy_variables!(model, master.linking_structure)

function BendersX.evaluate_objective(
    master::RenamedFieldMaster,
    linking_vars::AbstractVector{<:Real},
    auxiliary_vars::AbstractVector{<:Real},
)
    return dot(master.linking_costs, linking_vars) +
           dot(master.auxiliary_costs, auxiliary_vars)
end

function BendersX.add_cuts!(
    master::RenamedFieldMaster,
    hyperplanes::Vector{BendersX.Hyperplane},
)
    master.cut_batches += 1
    cuts = BendersX.hyperplanes_to_expression(
        master.jump_model,
        hyperplanes,
        master.linking_vars,
        master.auxiliary_vars,
    )
    return @constraint(master.jump_model, 0.0 .>= cuts)
end

function renamed_field_master(; optimizer = GLPK.Optimizer)
    provided = Master(
        MasterInterfaceTestData();
        model = build_master_interface_test_model!,
        optimizer = optimizer,
    )
    return RenamedFieldMaster(
        provided.model,
        provided.x_tuple,
        provided.x,
        provided.t,
        provided.c_x,
        provided.c_t,
        0,
    )
end

struct IncompleteInterfaceMaster <: BendersX.AbstractMaster end

@testset "AbstractMaster interface" begin
    @testset "missing methods fail through the documented interface" begin
        master = IncompleteInterfaceMaster()
        @test_throws BendersX.UnimplementedInterfaceException BendersX.master_model(master)
        @test_throws BendersX.UnimplementedInterfaceException BendersX.linking_variables(master)
        @test_throws BendersX.UnimplementedInterfaceException BendersX.auxiliary_variables(master)
        @test_throws BendersX.UnimplementedInterfaceException BendersX.copy_linking_variable_tuple!(Model(), master)
        @test_throws BendersX.UnimplementedInterfaceException BendersX.evaluate_objective(
            master,
            Float64[],
            Float64[],
        )
        @test_throws BendersX.UnimplementedInterfaceException BendersX.add_cuts!(
            master,
            BendersX.Hyperplane[],
        )
    end

    @testset "provided Master implements the public interface" begin
        master = Master(
            MasterInterfaceTestData();
            model = build_master_interface_test_model!,
            optimizer = GLPK.Optimizer,
        )

        @test BendersX.master_model(master) === master.model
        @test BendersX.linking_variables(master) === master.x
        @test BendersX.auxiliary_variables(master) === master.t
        @test BendersX.evaluate_objective(master, [0.5], [1.5]) == 2.0
        @test_throws DimensionMismatch BendersX.evaluate_objective(master, Float64[], [1.5])
        @test_throws DimensionMismatch BendersX.evaluate_objective(master, [0.5], Float64[])
        @test_throws DimensionMismatch BendersX.infeasibility_report(master, Float64[], [1.5])
        @test_throws DimensionMismatch BendersX.infeasibility_report(master, [0.5], Float64[])
    end

    @testset "different field layout supports built-in components" begin
        master = renamed_field_master()
        copied_model = Model()
        copied = BendersX.copy_linking_variable_tuple!(copied_model, master)

        @test keys(copied) == (:open,)
        @test length(BendersX.var_from_tuple(copied)) == 1
        @test BendersX.evaluate_objective(master, [0.5], [1.5]) == 2.0

        first_oracle = ClassicalOracle(
            MasterInterfaceTestData(),
            master;
            model = build_master_interface_test_subproblem!,
            optimizer = GLPK.Optimizer,
        )
        second_oracle = ClassicalOracle(
            MasterInterfaceTestData(),
            master;
            model = build_master_interface_test_subproblem!,
            optimizer = GLPK.Optimizer,
        )
        unified_oracle = UnifiedOracle(
            MasterInterfaceTestData(),
            master;
            model = build_master_interface_test_subproblem!,
            optimizer = GLPK.Optimizer,
        )
        pareto_oracle = ParetoOracle(
            MasterInterfaceTestData(),
            master,
            ParetoOracleParam([0.5]);
            model = build_master_interface_test_subproblem!,
            optimizer = GLPK.Optimizer,
        )

        unified_is_in_L, unified_cuts, unified_objectives = BendersX.generate_cuts(
            unified_oracle,
            [0.0],
            [0.0],
        )
        @test !unified_is_in_L
        @test length(unified_cuts) == 1
        @test length(only(unified_cuts).a_x) == 1
        @test length(only(unified_cuts).a_t) == 1
        @test isinf(only(unified_objectives))

        pareto_is_in_L, pareto_cuts, pareto_objectives = BendersX.generate_cuts(
            pareto_oracle,
            [0.0],
            [0.0],
        )
        @test !pareto_is_in_L
        @test length(pareto_cuts) == 1
        @test length(only(pareto_cuts).a_x) == 1
        @test length(only(pareto_cuts).a_t) == 1
        @test isapprox(only(pareto_objectives), 2.0; atol = 1.0e-8)

        separable = SeparableOracle(master, [first_oracle])
        @test separable.dim_global_auxiliary == 1

        split = SplitOracle(
            master,
            (first_oracle, second_oracle);
            param = SplitOracleParam(
                dcglp_param = DcglpParam(verbose = false),
            ),
        )
        @test length(split.dcglp[:sx]) == 1
        @test length(split.dcglp[:st]) == 1
    end

    @testset "branch-and-bound environment uses the new master interface" begin
        if MASTER_INTERFACE_CPLEX_AVAILABLE
            master = renamed_field_master(; optimizer = CPLEX.Optimizer)
            set_integer(only(BendersX.linking_variables(master)))
            oracle = ClassicalOracle(
                MasterInterfaceTestData(),
                master;
                model = build_master_interface_test_subproblem!,
                param = ClassicalOracleParam(atol = 1.0e-8),
                optimizer = GLPK.Optimizer,
            )

            bnb = BendersBnB(
                master,
                oracle;
                param = BendersBnBParam(
                    time_limit = 30.0,
                    gap_tolerance = 1.0e-8,
                    verbose = false,
                ),
            )
            result = solve!(bnb)

            @test bnb.termination_status isa Optimal
            @test isapprox(bnb.obj_value, 2.0; atol = 1.0e-8)
            @test only(result.n_lazy_cuts) >= 1
            @test !isempty(result)
        else
            @test_skip "CPLEX is required for the BendersBnB integration test"
        end
    end

    @testset "sequential environment delegates cut insertion" begin
        master = renamed_field_master()
        oracle = ClassicalOracle(
            MasterInterfaceTestData(),
            master;
            model = build_master_interface_test_subproblem!,
            optimizer = GLPK.Optimizer,
        )
        env = BendersSeq(
            master,
            oracle;
            param = BendersSeqParam(
                time_limit = 30.0,
                gap_tolerance = 1.0e-8,
                verbose = false,
            ),
        )

        result = solve!(env)

        @test env.termination_status isa Optimal
        @test isapprox(env.obj_value, 2.0; atol = 1.0e-8)
        @test master.cut_batches >= 1
        @test !isempty(result)
    end

    @testset "in-out environment uses the same master interface" begin
        master = renamed_field_master()
        oracle = ClassicalOracle(
            MasterInterfaceTestData(),
            master;
            model = build_master_interface_test_subproblem!,
            optimizer = GLPK.Optimizer,
        )
        env = BendersSeqInOut(
            master,
            oracle;
            param = BendersSeqInOutParam(
                stabilizing_x = [0.0],
                time_limit = 30.0,
                gap_tolerance = 1.0e-8,
                verbose = false,
            ),
        )

        result = solve!(env)

        @test env.termination_status isa Optimal
        @test isapprox(env.obj_value, 2.0; atol = 1.0e-8)
        @test master.cut_batches >= 1
        @test !isempty(result)
    end
end
