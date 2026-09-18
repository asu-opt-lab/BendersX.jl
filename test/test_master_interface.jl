using Test
using JuMP
using GLPK
using LinearAlgebra

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
    first_stage_variables::Vector{VariableRef}
    recourse_variables::Vector{VariableRef}
    first_stage_costs::Vector{Float64}
    recourse_costs::Vector{Float64}
    cut_batches::Int
end

BendersX.master_model(master::RenamedFieldMaster) = master.jump_model
BendersX.linking_variables(master::RenamedFieldMaster) = master.first_stage_variables
BendersX.auxiliary_variables(master::RenamedFieldMaster) = master.recourse_variables
BendersX.copy_linking_variables!(model::Model, master::RenamedFieldMaster) =
    BendersX.copy_variables!(model, master.linking_structure)

function BendersX.evaluate_primal_objective(
    master::RenamedFieldMaster,
    x_value::AbstractVector{<:Real},
    recourse_value::AbstractVector{<:Real},
)
    return dot(master.first_stage_costs, x_value) +
           dot(master.recourse_costs, recourse_value)
end

function BendersX.add_cuts!(
    master::RenamedFieldMaster,
    hyperplanes::Vector{BendersX.Hyperplane},
)
    master.cut_batches += 1
    cuts = BendersX.hyperplanes_to_expression(
        master.jump_model,
        hyperplanes,
        master.first_stage_variables,
        master.recourse_variables,
    )
    return @constraint(master.jump_model, 0.0 .>= cuts)
end

function renamed_field_master()
    provided = Master(
        MasterInterfaceTestData();
        model = build_master_interface_test_model!,
        optimizer = GLPK.Optimizer,
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
        @test_throws BendersX.UnimplementedInterfaceException BendersX.copy_linking_variables!(Model(), master)
        @test_throws BendersX.UnimplementedInterfaceException BendersX.evaluate_primal_objective(
            master,
            Float64[],
            Float64[],
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
        @test BendersX.evaluate_primal_objective(master, [0.5], [1.5]) == 2.0
        @test_throws DimensionMismatch BendersX.evaluate_primal_objective(master, Float64[], [1.5])
        @test_throws DimensionMismatch BendersX.evaluate_primal_objective(master, [0.5], Float64[])
        @test_throws DimensionMismatch BendersX.infeasibility_report(master, Float64[], [1.5])
        @test_throws DimensionMismatch BendersX.infeasibility_report(master, [0.5], Float64[])
    end

    @testset "different field layout supports built-in components" begin
        master = renamed_field_master()
        copied_model = Model()
        copied = BendersX.copy_linking_variables!(copied_model, master)

        @test fieldnames(RenamedFieldMaster) == (
            :jump_model,
            :linking_structure,
            :first_stage_variables,
            :recourse_variables,
            :first_stage_costs,
            :recourse_costs,
            :cut_batches,
        )
        @test keys(copied) == (:open,)
        @test length(BendersX.var_from_tuple(copied)) == 1
        @test BendersX.evaluate_primal_objective(master, [0.5], [1.5]) == 2.0

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

        @test unified_oracle isa UnifiedOracle
        @test pareto_oracle isa ParetoOracle

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

        bnb = BendersBnB(
            master,
            first_oracle;
            param = BendersBnBParam(verbose = false),
        )
        @test bnb.master === master
        @test applicable(
            BendersX.lazy_callback,
            nothing,
            master,
            BendersX.BendersBnBLog(),
            bnb.param,
            bnb.lazy_callback,
        )
        @test applicable(
            BendersX.user_callback,
            nothing,
            master,
            BendersX.BendersBnBLog(),
            bnb.param,
            UserCallback(unified_oracle),
        )
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
