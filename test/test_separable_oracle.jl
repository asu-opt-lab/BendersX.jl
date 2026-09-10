using Test
using BendersX
using JuMP
using HiGHS
using MathOptInterface

const SO_MOI = MathOptInterface

struct SeparableOracleTestData
    n_scenarios::Int
    costs::Matrix{Float64}
end

function separable_oracle_optimizer()
    return optimizer_with_attributes(HiGHS.Optimizer, SO_MOI.Silent() => true)
end

function update_separable_oracle_master!(model::Model, data::SeparableOracleTestData)
    @variable(model, x[1:2], Bin)
    @variable(model, t[1:data.n_scenarios] >= 0)
    @constraint(model, sum(x) >= 1)
    @objective(model, Min, 0.2 * sum(x) + sum(t))
    return (x = x,), t
end

function update_block_ufl_master!(model::Model, data::UFLPData)
    @variable(model, x[1:data.n_facilities], Bin)
    @variable(model, t[1:(2 * data.n_customers)] >= 0)
    @constraint(model, sum(x) >= 1)
    @objective(model, Min, data.fixed_costs' * x + sum(t))
    return (x = x,), t
end

function update_separable_oracle_sub!(
    model::Model,
    data::SeparableOracleTestData,
    scen_idx::Int;
    x,
)
    @variable(model, y[1:2] >= 0)
    @constraint(model, sum(y) == 1)
    @constraint(model, [i in 1:2], y[i] <= x[i])
    @objective(model, Min, sum(data.costs[i, scen_idx] * y[i] for i in 1:2))
    return nothing
end

function separable_oracle_fixture()
    data = SeparableOracleTestData(2, [1.0 3.0; 3.0 1.0])
    master = Master(
        data;
        model = update_separable_oracle_master!,
        optimizer = separable_oracle_optimizer(),
    )
    return data, master
end

function separable_split_param(; reuse_dcglp::Bool = true)
    return SplitOracleParam(;
        dcglp_param = DcglpParam(;
            optimizer = separable_oracle_optimizer(),
            time_limit = 20.0,
            gap_tolerance = 1.0e-6,
            halt_limit = 3,
            iter_limit = 20,
            verbose = false,
        ),
        split_index_selection_rule = MostFractional(),
        disjunctive_cut_append_rule = AllDisjunctiveCuts(),
        add_benders_cuts_to_master = true,
        reuse_dcglp = reuse_dcglp,
        strengthened = false,
        lift = false,
    )
end

mutable struct StoredLocalDisjunctiveOracle <: BendersX.AbstractDisjunctiveOracle
    cut::BendersX.Hyperplane
    objective::Float64
end

BendersX.auxiliary_dimension(::StoredLocalDisjunctiveOracle) = 1

function BendersX.generate_cuts(
    oracle::StoredLocalDisjunctiveOracle,
    ::Vector{Float64},
    ::Vector{Float64};
    tol_normalize = 1.0,
    time_limit = 3600.0,
)
    return false, [oracle.cut], [oracle.objective]
end

struct HomogeneousDisjunctiveOracle <: BendersX.AbstractDisjunctiveOracle
    scen_idx::Int
end

BendersX.auxiliary_dimension(::HomogeneousDisjunctiveOracle) = 1

function HomogeneousDisjunctiveOracle(
    data,
    master::BendersX.AbstractMaster;
    model = update_sub_model!,
    scen_idx::Int,
    param::BendersX.AbstractOracleParam,
    optimizer = nothing,
)
    return HomogeneousDisjunctiveOracle(scen_idx)
end

function BendersX.generate_cuts(
    oracle::HomogeneousDisjunctiveOracle,
    x_value::Vector{Float64},
    ::Vector{Float64};
    tol_normalize = 1.0,
    time_limit = 3600.0,
)
    cut = BendersX.Hyperplane(length(x_value), 1)
    cut.a_t[1] = -1.0
    cut.a_0 = Float64(oracle.scen_idx)
    return false, [cut], [Float64(oracle.scen_idx)]
end

struct SeparableContractOracle <: BendersX.AbstractTypicalOracle
    a_x_length::Int
    a_t_length::Int
    objective_count::Int
    is_in_L::Bool
end

BendersX.auxiliary_dimension(::SeparableContractOracle) = 1

struct AuxiliaryDimensionTestOracle <: BendersX.AbstractTypicalOracle
    dimension::Int
end

BendersX.auxiliary_dimension(oracle::AuxiliaryDimensionTestOracle) =
    oracle.dimension

function BendersX.generate_cuts(
    oracle::SeparableContractOracle,
    ::Vector{Float64},
    ::Vector{Float64};
    tol_normalize = 1.0,
    time_limit = 3600.0,
)
    cut = BendersX.Hyperplane(oracle.a_x_length, oracle.a_t_length)
    return oracle.is_in_L, [cut], fill(7.0, oracle.objective_count)
end

function local_split_oracle(data, master, scen_idx::Int; reuse_dcglp::Bool = true)
    typical_pair = ntuple(2) do _
        ClassicalOracle(
            data,
            master;
            model = update_separable_oracle_sub!,
            scen_idx = scen_idx,
            optimizer = separable_oracle_optimizer(),
        )
    end
    return SplitOracle(
        master,
        typical_pair;
        param = separable_split_param(; reuse_dcglp = reuse_dcglp),
    )
end

@testset "SeparableOracle composition" begin
    @testset "accepts disjunctive children and embeds copies" begin
        _, master = separable_oracle_fixture()
        first_cut = BendersX.Hyperplane([1.0, 0.0], [-2.0], 3.0)
        second_cut = BendersX.Hyperplane([0.0, 1.0], [-4.0], 5.0)
        children = [
            StoredLocalDisjunctiveOracle(first_cut, 11.0),
            StoredLocalDisjunctiveOracle(second_cut, 13.0),
        ]
        oracle = SeparableOracle(master, children)

        is_in_L, cuts, objectives = BendersX.generate_cuts(
            oracle,
            [0.5, 0.5],
            [0.0, 0.0],
        )

        @test !is_in_L
        @test objectives == [11.0, 13.0]
        @test collect(cuts[1].a_t) == [-2.0, 0.0]
        @test collect(cuts[2].a_t) == [0.0, -4.0]
        @test cuts[1] !== first_cut
        @test cuts[2] !== second_cut
        @test collect(first_cut.a_t) == [-2.0]
        @test collect(second_cut.a_t) == [-4.0]

        BendersX.generate_cuts(oracle, [0.5, 0.5], [0.0, 0.0])
        @test length(first_cut.a_t) == 1
        @test length(second_cut.a_t) == 1
    end

    @testset "homogeneous constructor accepts disjunctive oracle types" begin
        data, master = separable_oracle_fixture()
        oracle = SeparableOracle(
            data,
            master,
            HomogeneousDisjunctiveOracle,
            data.n_scenarios;
            optimizer = nothing,
        )
        @test getfield.(oracle.oracles, :scen_idx) == [1, 2]
        _, cuts, objectives = BendersX.generate_cuts(oracle, [0.0, 0.0], [0.0, 0.0])
        @test objectives == [1.0, 2.0]
        @test collect(cuts[1].a_t) == [-1.0, 0.0]
        @test collect(cuts[2].a_t) == [0.0, -1.0]
    end

    @testset "supports multi-dimensional child auxiliary blocks" begin
        data = UFLPData(
            2,
            2,
            ones(2),
            zeros(2),
            [1.0 2.0; 3.0 4.0],
        )
        master = Master(
            data;
            model = update_block_ufl_master!,
            optimizer = separable_oracle_optimizer(),
        )
        oracle = SeparableOracle(
            master,
            [UFLKnapsackOracle(data), UFLKnapsackOracle(data)],
        )

        @test oracle.dim_auxiliary == 4
        @test oracle.auxiliary_ranges == [1:2, 3:4]
        @test BendersX.auxiliary_dimension(oracle) == 4

        is_in_L, cuts, objectives = BendersX.generate_cuts(
            oracle,
            [1.0, 0.0],
            zeros(4),
        )

        @test !is_in_L
        @test objectives == [1.0, 2.0, 1.0, 2.0]
        @test collect.(getfield.(cuts, :a_t)) == [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
        ]
    end

    @testset "validates child and global output dimensions" begin
        _, master = separable_oracle_fixture()
        valid = SeparableContractOracle(2, 1, 1, true)
        oracle = SeparableOracle(master, [valid, valid])
        is_in_L, _, objectives = BendersX.generate_cuts(
            oracle,
            [0.0, 0.0],
            [0.0, 0.0],
        )
        @test is_in_L
        @test objectives == [7.0, 7.0]
        @test_throws DimensionMismatch BendersX.generate_cuts(
            oracle,
            [0.0, 0.0],
            [0.0],
        )

        bad_t = SeparableOracle(master, [SeparableContractOracle(2, 2, 1, false), valid])
        @test_throws DimensionMismatch BendersX.generate_cuts(
            bad_t,
            [0.0, 0.0],
            [0.0, 0.0],
        )

        bad_x = SeparableOracle(master, [SeparableContractOracle(1, 1, 1, false), valid])
        @test_throws DimensionMismatch BendersX.generate_cuts(
            bad_x,
            [0.0, 0.0],
            [0.0, 0.0],
        )

        bad_objective = SeparableOracle(master, [SeparableContractOracle(2, 1, 2, false), valid])
        @test_throws DimensionMismatch BendersX.generate_cuts(
            bad_objective,
            [0.0, 0.0],
            [0.0, 0.0],
        )
    end

    @testset "local SplitOracles reuse one-dimensional cut history" begin
        data, master = separable_oracle_fixture()
        split_children = [local_split_oracle(data, master, j) for j in 1:2]
        oracle = SeparableOracle(master, split_children)

        for _ in 1:2
            is_in_L, cuts, objectives = BendersX.generate_cuts(
                oracle,
                [0.5, 0.5],
                [0.0, 0.0];
                time_limit = 20.0,
            )
            @test is_in_L isa Bool
            @test all(length(cut.a_t) == 2 for cut in cuts)
            @test length(objectives) == 2
            @test all(
                length(cut.a_t) == 1
                for child in split_children for cut in child.disjunctive_cuts
            )
        end
    end

    @testset "reverse-polar normalization uses the local t dimension" begin
        data, master = separable_oracle_fixture()
        typical_pair = ntuple(2) do _
            ClassicalOracle(
                data,
                master;
                model = update_separable_oracle_sub!,
                scen_idx = 1,
                optimizer = separable_oracle_optimizer(),
            )
        end
        normalization = ReversePolarNormalization()
        split = SplitOracle(
            master,
            typical_pair;
            normalization = normalization,
            param = SplitOracleParam(;
                dcglp_param = separable_split_param().dcglp_param,
            ),
        )

        @test split.dim_auxiliary == 1
        @test BendersX.auxiliary_dimension(split) == 1
        @test length(split.dcglp[:st]) == 1
        @test normalization.core_direction_x == zeros(master.dim_x)
        @test normalization.core_direction_t == ones(1)

        @test_throws DimensionMismatch SplitOracle(
            master,
            typical_pair;
            normalization = ReversePolarNormalization(;
                core_direction_x = zeros(master.dim_x),
                core_direction_t = ones(master.dim_t),
            ),
            param = SplitOracleParam(;
                dcglp_param = separable_split_param().dcglp_param,
            ),
        )
    end

    @testset "global SplitOracle still accepts separable typical components" begin
        data, master = separable_oracle_fixture()
        separable_pair = ntuple(2) do _
            children = [
                ClassicalOracle(
                    data,
                    master;
                    model = update_separable_oracle_sub!,
                    scen_idx = j,
                    optimizer = separable_oracle_optimizer(),
                ) for j in 1:2
            ]
            SeparableOracle(master, children)
        end
        split = SplitOracle(
            master,
            separable_pair;
            param = separable_split_param(; reuse_dcglp = false),
        )
        is_in_L, cuts, objectives = BendersX.generate_cuts(
            split,
            [0.5, 0.5],
            [0.0, 0.0];
            time_limit = 20.0,
        )
        @test is_in_L isa Bool
        @test all(length(cut.a_t) == 2 for cut in cuts)
        @test length(objectives) == 2
    end

    @testset "typical classification and auxiliary dimensions" begin
        data, master = separable_oracle_fixture()
        typical = SeparableOracle(master, [
            ClassicalOracle(
                data,
                master;
                model = update_separable_oracle_sub!,
                scen_idx = j,
                optimizer = separable_oracle_optimizer(),
            ) for j in 1:2
        ])
        disjunctive = SeparableOracle(master, [
            StoredLocalDisjunctiveOracle(BendersX.Hyperplane(2, 1), 0.0)
            for _ in 1:2
        ])
        mixed = SeparableOracle(master, BendersX.AbstractOracle[
            typical.oracles[1],
            disjunctive.oracles[2],
        ])

        @test BendersX.is_typical_oracle(typical)
        @test !BendersX.is_typical_oracle(disjunctive)
        @test !BendersX.is_typical_oracle(mixed)
        @test BendersX.auxiliary_dimension(first(typical.oracles)) == 1
        @test BendersX.auxiliary_dimension(typical) == 2

        ufl_data = UFLPData(
            2,
            3,
            ones(3),
            ones(2),
            ones(2, 3),
        )
        @test BendersX.auxiliary_dimension(UFLKnapsackOracle(ufl_data)) == 3

        @test SplitOracle(master, (typical, typical)) isa SplitOracle
        @test_throws ArgumentError SplitOracle(master, (mixed, typical))
        @test_throws DimensionMismatch SplitOracle(
            master,
            (AuxiliaryDimensionTestOracle(1), AuxiliaryDimensionTestOracle(2)),
        )
        @test_throws ArgumentError SplitOracle(
            master,
            (AuxiliaryDimensionTestOracle(0), AuxiliaryDimensionTestOracle(0)),
        )
        @test_throws DimensionMismatch SeparableOracle(
            master,
            [AuxiliaryDimensionTestOracle(2), AuxiliaryDimensionTestOracle(2)],
        )
    end

    @testset "two-scenario BendersSeq matches extensive form" begin
        data, master = separable_oracle_fixture()
        oracle = SeparableOracle(master, [
            local_split_oracle(data, master, j; reuse_dcglp = true) for j in 1:2
        ])
        env = BendersSeq(
            master,
            oracle;
            param = BendersSeqParam(
                time_limit = 30.0,
                gap_tolerance = 1.0e-6,
                verbose = false,
            ),
        )
        solve!(env)

        extensive = Model(separable_oracle_optimizer())
        @variable(extensive, x[1:2], Bin)
        @variable(extensive, y[1:2, 1:2] >= 0)
        @constraint(extensive, sum(x) >= 1)
        @constraint(extensive, [j in 1:2], sum(y[:, j]) == 1)
        @constraint(extensive, [i in 1:2, j in 1:2], y[i, j] <= x[i])
        @objective(
            extensive,
            Min,
            0.2 * sum(x) + sum(data.costs[i, j] * y[i, j] for i in 1:2, j in 1:2),
        )
        optimize!(extensive)

        @test env.termination_status == Optimal()
        @test termination_status(extensive) == SO_MOI.OPTIMAL
        @test isapprox(env.obj_value, objective_value(extensive); atol = 1.0e-5)
    end
end
