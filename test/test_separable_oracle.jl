using Test
using BendersX
using JuMP
using HiGHS
using MathOptInterface
using SparseArrays

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
    subproblem_idx::Int;
    x,
)
    @variable(model, y[1:2] >= 0)
    @constraint(model, sum(y) == 1)
    @constraint(model, [i in 1:2], y[i] <= x[i])
    @objective(model, Min, sum(data.costs[i, subproblem_idx] * y[i] for i in 1:2))
    return nothing
end

function separable_oracle_fixture(n_scenarios::Int = 2)
    costs = zeros(2, n_scenarios)
    for j in 1:n_scenarios
        costs[:, j] = isodd(j) ? [1.0, 3.0] : [3.0, 1.0]
    end
    data = SeparableOracleTestData(n_scenarios, costs)
    master = Master(
        data;
        model = update_separable_oracle_master!,
        optimizer = separable_oracle_optimizer(),
    )
    return data, master
end

function classical_separable_component(
    data,
    master,
    subproblem_indices::Vector{Int},
)
    children = [
        ClassicalOracle(
            data,
            master;
            model = update_separable_oracle_sub!,
            subproblem_idx = subproblem_idx,
            optimizer = separable_oracle_optimizer(),
        ) for subproblem_idx in subproblem_indices
    ]
    return SeparableOracle(
        master,
        children;
        subproblem_indices = subproblem_indices,
        auxiliary_indices = [[subproblem_idx] for subproblem_idx in subproblem_indices],
    )
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

mutable struct StoredLocalTypicalOracle <: BendersX.AbstractTypicalOracle
    dimension::Int
    cut::BendersX.Hyperplane
    objectives::Vector{Float64}
    received_t::Vector{Vector{Float64}}
end

BendersX.auxiliary_dimension(oracle::StoredLocalTypicalOracle) = oracle.dimension

function BendersX.generate_cuts(
    oracle::StoredLocalTypicalOracle,
    ::Vector{Float64},
    t_value::Vector{Float64};
    tol_normalize = 1.0,
    time_limit = 3600.0,
)
    push!(oracle.received_t, copy(t_value))
    return false, [deepcopy(oracle.cut)], copy(oracle.objectives)
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
    subproblem_idx::Int
end

BendersX.auxiliary_dimension(::HomogeneousDisjunctiveOracle) = 1

function HomogeneousDisjunctiveOracle(
    data,
    master::BendersX.AbstractMaster;
    model = update_sub_model!,
    subproblem_idx::Int,
    param::BendersX.AbstractOracleParam,
    optimizer = nothing,
)
    return HomogeneousDisjunctiveOracle(subproblem_idx)
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
    cut.a_0 = Float64(oracle.subproblem_idx)
    return false, [cut], [Float64(oracle.subproblem_idx)]
end

struct SeparableContractOracle <: BendersX.AbstractTypicalOracle
    a_x_length::Int
    a_t_length::Int
    objective_count::Int
    is_in_L::Bool
end

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

function local_split_oracle(data, master, subproblem_idx::Int; reuse_dcglp::Bool = true)
    typical_pair = ntuple(2) do _
        classical_separable_component(data, master, [subproblem_idx])
    end
    return SplitOracle(
        master,
        typical_pair;
        param = separable_split_param(; reuse_dcglp = reuse_dcglp),
    )
end

function grouped_split_oracle(data, master, subproblem_indices::Vector{Int})
    typical_pair = ntuple(2) do _
        classical_separable_component(data, master, subproblem_indices)
    end
    return SplitOracle(
        master,
        typical_pair;
        param = separable_split_param(; reuse_dcglp = false),
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

        @test oracle.subproblem_indices == [[1], [2]]
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
        @test getfield.(oracle.oracles, :subproblem_idx) == [1, 2]
        @test oracle.subproblem_indices == [[1], [2]]
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
            [UFLKnapsackOracle(data), UFLKnapsackOracle(data)];
            auxiliary_indices = [[1, 2], [3, 4]],
        )

        @test oracle.dim_auxiliary == 4
        @test oracle.subproblem_indices == [[1], [2]]
        @test oracle.auxiliary_indices == [[1, 2], [3, 4]]
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

    @testset "infers auxiliary indices from subproblem indices" begin
        _, master = separable_oracle_fixture(10)
        selected_indices = [2, 3, 10]
        children = [
            StoredLocalTypicalOracle(
                1,
                BendersX.Hyperplane(
                    [1.0, 0.0],
                    [-Float64(index)],
                    4.0,
                ),
                [100.0 + index],
                Vector{Float64}[],
            ) for index in selected_indices
        ]
        oracle = SeparableOracle(
            master,
            children;
            subproblem_indices = selected_indices,
        )

        is_in_L, cuts, objectives = BendersX.generate_cuts(
            oracle,
            [0.5, 0.5],
            collect(1.0:10.0),
        )
        @test !is_in_L
        @test getfield.(children, :received_t) == [[[2.0]], [[3.0]], [[10.0]]]
        @test objectives == [102.0, 103.0, 110.0]
        @test findnz.(getfield.(cuts, :a_t)) == [
            ([2], [-2.0]),
            ([3], [-3.0]),
            ([10], [-10.0]),
        ]
        @test oracle.subproblem_indices == [[2], [3], [10]]
        @test oracle.auxiliary_indices == [[2], [3], [10]]
        @test oracle.dim_auxiliary == 3
        @test oracle.dim_global_auxiliary == 10
    end

    @testset "supports noncontiguous auxiliary indices for one component" begin
        _, master = separable_oracle_fixture(10)
        child = StoredLocalTypicalOracle(
            3,
            BendersX.Hyperplane(
                [1.0, 0.0],
                [-8.0, -2.0, -5.0],
                4.0,
            ),
            [108.0, 102.0, 105.0],
            Vector{Float64}[],
        )
        oracle = SeparableOracle(
            master,
            [child];
            subproblem_indices = [[8, 2, 5]],
            auxiliary_indices = [[8, 2, 5]],
        )

        is_in_L, cuts, objectives = BendersX.generate_cuts(
            oracle,
            [0.5, 0.5],
            collect(1.0:10.0),
        )

        @test !is_in_L
        @test child.received_t == [[8.0, 2.0, 5.0]]
        @test objectives == [108.0, 102.0, 105.0]
        @test findnz(cuts[1].a_t) == ([2, 5, 8], [-2.0, -5.0, -8.0])
        @test oracle.auxiliary_indices == [[8, 2, 5]]

        typical_pair = ntuple(2) do _
            SeparableOracle(
                master,
                [AuxiliaryDimensionTestOracle(3)];
                subproblem_indices = [[8, 2, 5]],
                auxiliary_indices = [[8, 2, 5]],
            )
        end
        split = SplitOracle(
            master,
            typical_pair;
            param = separable_split_param(),
        )
        @test split.active_t_indices == [8, 2, 5]
    end

    @testset "groups subproblem indices by component for nested composition" begin
        data, master = separable_oracle_fixture(4)
        first_group = classical_separable_component(data, master, [1, 2])
        second_group = classical_separable_component(data, master, [3, 4])
        oracle = SeparableOracle(
            master,
            [first_group, second_group];
            subproblem_indices = [[1, 2], [3, 4]],
            auxiliary_indices = [[1, 2], [3, 4]],
        )

        @test oracle.subproblem_indices == [[1, 2], [3, 4]]
        @test oracle.auxiliary_indices == [[1, 2], [3, 4]]
        @test oracle.dim_auxiliary == 4
        @test oracle.dim_global_auxiliary == 4

        is_in_L, cuts, objectives = BendersX.generate_cuts(
            oracle,
            [1.0, 0.0],
            zeros(4),
        )
        @test !is_in_L
        @test objectives == [1.0, 3.0, 1.0, 3.0]
        @test all(length(cut.a_t) == 4 for cut in cuts)
        @test findnz.(getfield.(cuts, :a_t)) == [
            ([1], [-1.0]),
            ([2], [-1.0]),
            ([3], [-1.0]),
            ([4], [-1.0]),
        ]
    end

    @testset "validates grouped SeparableOracle mappings" begin
        _, master = separable_oracle_fixture(4)
        two_dimensional = AuxiliaryDimensionTestOracle(2)

        @test_throws DimensionMismatch SeparableOracle(
            master,
            [two_dimensional, two_dimensional];
            subproblem_indices = [[1, 2]],
            auxiliary_indices = [[1, 2], [3, 4]],
        )
        @test_throws ArgumentError SeparableOracle(
            master,
            [two_dimensional, two_dimensional];
            subproblem_indices = [[1, 2], [2, 3]],
            auxiliary_indices = [[1, 2], [3, 4]],
        )
        @test_throws DimensionMismatch SeparableOracle(
            master,
            [two_dimensional, two_dimensional];
            subproblem_indices = [[1, 2], [3, 4]],
            auxiliary_indices = [[1], [2, 3, 4]],
        )
        @test_throws ArgumentError SeparableOracle(
            master,
            [two_dimensional];
            subproblem_indices = [[1, 2]],
        )
        @test_throws DimensionMismatch SeparableOracle(
            master,
            [two_dimensional];
            subproblem_indices = [1],
        )
        @test_throws ArgumentError SeparableOracle(
            master,
            [AuxiliaryDimensionTestOracle(1)];
            subproblem_indices = [5],
        )
        @test_throws ArgumentError SeparableOracle(
            master,
            [two_dimensional, two_dimensional];
            subproblem_indices = [[1, 2], [3, 4]],
            auxiliary_indices = [[1, 2], [2, 3]],
        )
        @test_throws ArgumentError SeparableOracle(
            master,
            [two_dimensional, two_dimensional];
            subproblem_indices = [[1, 2], [3, 4]],
            auxiliary_indices = [[1, 2], [3, 5]],
        )
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

    @testset "subset SplitOracles reuse global cut history" begin
        data, master = separable_oracle_fixture()
        split_children = [local_split_oracle(data, master, j) for j in 1:2]
        oracle = SeparableOracle(master, split_children)

        @test getfield.(split_children, :active_t_indices) == [[1], [2]]
        @test length.(getindex.(getfield.(split_children, :dcglp), Ref(:st))) == [2, 2]
        for _ in 1:2
            is_in_L, cuts, objectives = BendersX.generate_cuts(
                oracle,
                [0.5, 0.5],
                [0.0, 0.0];
                time_limit = 20.0,
            )
            @test is_in_L isa Bool
            @test all(length(cut.a_t) == master.dim_t for cut in cuts)
            @test length(objectives) == 2
            @test all(
                length(cut.a_t) == master.dim_t
                for child in split_children for cut in child.disjunctive_cuts
            )
        end
    end

    @testset "SplitOracle keeps selected blocks in the global auxiliary space" begin
        data, master = separable_oracle_fixture(4)
        split = grouped_split_oracle(data, master, [2, 4])

        @test split.dim_auxiliary == 2
        @test split.active_t_indices == [2, 4]
        @test length(split.dcglp[:st]) == 4
        @test_throws DimensionMismatch BendersX.generate_cuts(
            split,
            [0.5, 0.5],
            zeros(2),
        )

        is_in_L, cuts, objectives = BendersX.generate_cuts(
            split,
            [0.5, 0.5],
            zeros(4);
            time_limit = 20.0,
        )
        @test is_in_L isa Bool
        @test length(objectives) == 2
        @test all(
            length(cut.a_t) == master.dim_t for cut in cuts
        )
        @test all(
            length(cut.a_t) == master.dim_t for cut in split.disjunctive_cuts
        )
    end

    @testset "SplitOracle validates component mappings" begin
        _, master = separable_oracle_fixture(4)
        left = SeparableOracle(
            master,
            [AuxiliaryDimensionTestOracle(1)];
            subproblem_indices = [2],
            auxiliary_indices = [[2]],
        )
        different_index = SeparableOracle(
            master,
            [AuxiliaryDimensionTestOracle(1)];
            subproblem_indices = [3],
            auxiliary_indices = [[2]],
        )
        different_mapping = SeparableOracle(
            master,
            [AuxiliaryDimensionTestOracle(1)];
            subproblem_indices = [2],
            auxiliary_indices = [[3]],
        )

        index_error = try
            SplitOracle(master, (left, different_index))
            nothing
        catch err
            err
        end
        @test index_error isa ArgumentError
        @test occursin("same subproblems", sprint(showerror, index_error))

        mapping_error = try
            SplitOracle(master, (left, different_mapping))
            nothing
        catch err
            err
        end
        @test mapping_error isa DimensionMismatch
        @test occursin("same auxiliary_indices", sprint(showerror, mapping_error))

        _, other_master = separable_oracle_fixture(5)
        different_global_dimension = SeparableOracle(
            other_master,
            [AuxiliaryDimensionTestOracle(1)];
            subproblem_indices = [2],
            auxiliary_indices = [[2]],
        )
        global_dimension_error = try
            SplitOracle(master, (left, different_global_dimension))
            nothing
        catch err
            err
        end
        @test global_dimension_error isa DimensionMismatch
        @test occursin(
            "global auxiliary dimension",
            sprint(showerror, global_dimension_error),
        )

        mixed_error = try
            SplitOracle(master, (left, AuxiliaryDimensionTestOracle(1)))
            nothing
        catch err
            err
        end
        @test mixed_error isa ArgumentError
        @test occursin("both be SeparableOracles", sprint(showerror, mixed_error))

        dimension_error = try
            SplitOracle(
                master,
                (AuxiliaryDimensionTestOracle(2), AuxiliaryDimensionTestOracle(2)),
            )
            nothing
        catch err
            err
        end
        @test dimension_error isa DimensionMismatch
        @test occursin("full auxiliary space", sprint(showerror, dimension_error))
    end

    @testset "nested SeparableOracle combines SplitOracle groups" begin
        data, master = separable_oracle_fixture(4)
        first_split = grouped_split_oracle(data, master, [1, 2])
        second_split = grouped_split_oracle(data, master, [3, 4])
        oracle = SeparableOracle(
            master,
            [first_split, second_split];
            subproblem_indices = [[1, 2], [3, 4]],
            auxiliary_indices = [[1, 2], [3, 4]],
        )

        @test oracle.subproblem_indices == [[1, 2], [3, 4]]
        @test vcat(first_split.active_t_indices, second_split.active_t_indices) == 1:4

        is_in_L, cuts, objectives = BendersX.generate_cuts(
            oracle,
            [0.5, 0.5],
            zeros(4);
            time_limit = 20.0,
        )
        @test is_in_L isa Bool
        @test length(objectives) == 4
        @test all(length(cut.a_t) == 4 for cut in cuts)

        @test_throws ArgumentError SeparableOracle(
            master,
            [first_split, grouped_split_oracle(data, master, [2, 4])];
            subproblem_indices = [[1, 2], [2, 4]],
            auxiliary_indices = [[1, 2], [2, 3]],
        )
    end

    @testset "reverse-polar normalization uses the global t dimension" begin
        data, master = separable_oracle_fixture()
        typical_pair = ntuple(2) do _
            SeparableOracle(
                master,
                [
                    ClassicalOracle(
                        data,
                        master;
                        model = update_separable_oracle_sub!,
                        subproblem_idx = 1,
                        optimizer = separable_oracle_optimizer(),
                    ),
                ];
                subproblem_indices = [1],
                auxiliary_indices = [[1]],
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
        @test length(split.dcglp[:st]) == master.dim_t
        @test split.active_t_indices == [1]
        @test normalization.core_direction_x == zeros(master.dim_x)
        @test normalization.core_direction_t == ones(master.dim_t)

        @test_throws DimensionMismatch SplitOracle(
            master,
            typical_pair;
            normalization = ReversePolarNormalization(;
                core_direction_x = zeros(master.dim_x),
                core_direction_t = ones(1),
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
                    subproblem_idx = j,
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
                subproblem_idx = j,
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
            (AuxiliaryDimensionTestOracle(1), AuxiliaryDimensionTestOracle(1)),
        )
        @test_throws DimensionMismatch SplitOracle(
            master,
            (AuxiliaryDimensionTestOracle(1), AuxiliaryDimensionTestOracle(2)),
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
