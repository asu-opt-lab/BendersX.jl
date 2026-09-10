using Test
using BendersX
import BendersX: copy_variables!, transfer_scaled_linear_rows_and_bounds_with_types!, UnimplementedInterfaceException
import MathOptInterface
using HiGHS
using JuMP

const MOI = MathOptInterface

struct SeparableInterruptionTestOracle <: BendersX.AbstractTypicalOracle
    interruption::Union{Nothing,Exception}
end

function BendersX.generate_cuts(
    oracle::SeparableInterruptionTestOracle,
    x_value::Vector{Float64},
    t_value::Vector{Float64};
    tol_normalize = 1.0,
    time_limit = 3600.0,
)
    !isnothing(oracle.interruption) && throw(oracle.interruption)
    return true, [BendersX.Hyperplane(length(x_value), length(t_value))], deepcopy(t_value)
end

@testset "BendersX copy_variables!" begin
    @testset "VariableRef" begin
        master_model = Model()
        @variable(master_model, u[1:10])
        @variable(master_model, v)
        x = (u = u, v = v)

        sub_model = Model()

        x_copy = copy_variables!(sub_model, x)

        function build_sub_model(model; u,v)
            @variable(model, y >= 0)
            @constraint(model, sum(u)+y == 1)
            @constraint(model, sum(v) == 1)
        end

        build_sub_model(sub_model; x_copy...)
        print(sub_model)
    end
    @testset "Array Container" begin
        @testset "1D Array" begin
            master_model = Model()
            @variable(master_model, u[1:10])
            @variable(master_model, v[1:2])
            x = (u = u, v = v)

            sub_model = Model()

            x_copy = copy_variables!(sub_model, x)

            function build_sub_model(model; u,v)
                @variable(model, y >= 0)
                @constraint(model, sum(u)+y == 1)
                @constraint(model, sum(v) == 1)
                @constraint(model, v[2]+y == 1)
            end

            build_sub_model(sub_model; x_copy...)
            print(sub_model)
        end
        @testset "2D Array" begin
            master_model = Model()
            @variable(master_model, u[1:10])
            @variable(master_model, v[1:3,1:4])
            x = (u = u, v = v)

            sub_model = Model()

            x_copy = copy_variables!(sub_model, x)

            function build_sub_model(model; u,v)
                @variable(model, y >= 0)
                @constraint(model, sum(u)+y == 1)
                @constraint(model, sum(v) == 1)
                @constraint(model, v[3,4]+y == 1)
            end

            build_sub_model(sub_model; x_copy...)
            print(sub_model)
        end
    end
    @testset "DenseAxisArray Container" begin
        @testset "1D Array" begin
            master_model = Model()
            @variable(master_model, u[1:10])
            @variable(master_model, v[2:3])
            x = (u = u, v = v)

            sub_model = Model()

            x_copy = copy_variables!(sub_model, x)

            function build_sub_model(model; u,v)
                @variable(model, y >= 0)
                @constraint(model, sum(u)+y == 1)
                @constraint(model, sum(v) == 1)
                @constraint(model, v[3]+y == 1)
            end

            build_sub_model(sub_model; x_copy...)
            print(sub_model)
        end
        @testset "1D Array with Symbol index" begin
            master_model = Model()
            @variable(master_model, u[1:10])
            @variable(master_model, v[[:A,:B]])
            x = (u = u, v = v)

            sub_model = Model()

            x_copy = copy_variables!(sub_model, x)

            function build_sub_model(model; u,v)
                @variable(model, y >= 0)
                @constraint(model, sum(u)+y == 1)
                @constraint(model, sum(v) == 1)
                @constraint(model, v[:A]+y == 1)
            end

            build_sub_model(sub_model; x_copy...)
            print(sub_model)
        end
        @testset "High dimensional array with Symbol index" begin
            master_model = Model()
            @variable(master_model, u[1:10])
            @variable(master_model, v[1:2, 2:3, [:A,:B], 4:5])
            x = (u = u, v = v)

            sub_model = Model()

            x_copy = copy_variables!(sub_model, x)

            function build_sub_model(model; u,v)
                @variable(model, y >= 0)
                @constraint(model, sum(u)+y == 1)
                @constraint(model, sum(v) == 1)
                @constraint(model, v[2,3,:A,4]+y == 1)
            end

            build_sub_model(sub_model; x_copy...)
            print(sub_model)
        end
    end
    @testset "SparseAxisArrays Container" begin
        @testset "Example 1" begin
            master_model = Model()
            @variable(master_model, u[1:10])
            @variable(master_model, v[i=1:2, j=i:2, k=j:4])
            x = (u = u, v = v)

            sub_model = Model()

            x_copy = copy_variables!(sub_model, x)

            function build_sub_model(model; u,v)
                @variable(model, y >= 0)
                @constraint(model, sum(u)+y == 1)
                @constraint(model, sum(v) == 1)
                @constraint(model, v[1,2,3]+y == 1)
            end

            build_sub_model(sub_model; x_copy...)
            print(sub_model)
        end
        @testset "Example 2" begin
            master_model = Model()
            @variable(master_model, u[1:10])
            @variable(master_model, v[i=1:4; mod(i, 2)==0])
            x = (u = u, v = v)

            sub_model = Model()

            x_copy = copy_variables!(sub_model, x)

            function build_sub_model(model; u,v)
                @variable(model, y >= 0)
                @constraint(model, sum(u)+y == 1)
                @constraint(model, sum(v) == 1)
                @constraint(model, v[2]+y == 1)
            end

            build_sub_model(sub_model; x_copy...)
            print(sub_model)
        end
    end
end

@testset "Default GLPK optimizer attachment" begin
    data = CFLPData(
        2,
        2,
        [2.0, 2.0],
        [1.0, 1.0],
        [3.0, 4.0],
        [1.0 2.0; 2.0 1.0],
    )

    master = Master(data)
    classical = ClassicalOracle(data, master)
    unified = UnifiedOracle(data, master)
    pareto = ParetoOracle(data, master, ParetoOracleParam(fill(0.5, data.n_facilities)))
    knapsack = CFLKnapsackOracle(data, master)

    for model in (master.model, classical.model, unified.model, pareto.model, pareto.pareto_model, knapsack.model)
        @test occursin("GLPK", solver_name(model))
    end
end

@testset "Invalid optimizer errors include model context" begin
    data = CFLPData(
        2,
        2,
        [2.0, 2.0],
        [1.0, 1.0],
        [3.0, 4.0],
        [1.0 2.0; 2.0 1.0],
    )

    err = try
        Master(data; optimizer = nothing)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("Master model could not attach optimizer nothing", sprint(showerror, err))
    @test occursin("Omit `optimizer` to use the default GLPK optimizer", sprint(showerror, err))

    err = try
        Master(data; optimizer = :not_an_optimizer)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("Master model could not attach optimizer :not_an_optimizer", sprint(showerror, err))
    @test occursin("Original error:", sprint(showerror, err))

    master = Master(data)
    err = try
        ClassicalOracle(data, master; optimizer = nothing)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("ClassicalOracle subproblem model could not attach optimizer nothing", sprint(showerror, err))
end

@testset "Default optimizer is attached before model-update functions run" begin
    struct AttrData end
    data = AttrData()

    function update_master_model!(model::Model, data::AttrData)
        set_optimizer_attribute(model, MOI.Silent(), true)
        @variable(model, x[1:2], Bin)
        @variable(model, t >= 0)
        @objective(model, Min, sum(x) + t)
        return (x = x,), t
    end

    function update_sub_model!(model::Model, data::AttrData, scen_idx::Int; x)
        set_optimizer_attribute(model, MOI.Silent(), true)
        @variable(model, y >= 0)
        @objective(model, Min, y)
        @constraint(model, y >= 1 - x[1])
        return nothing
    end

    master = Master(data; model = update_master_model!)
    oracle = ClassicalOracle(data, master; model = update_sub_model!)

    @test occursin("GLPK", solver_name(master.model))
    @test occursin("GLPK", solver_name(oracle.model))
end

@testset "model keyword, unconstrained data types, and revised constructors" begin
    struct ModelKeywordData
        n_facilities::Int
        n_customers::Int
        capacities::Vector{Float64}
        demands::Vector{Float64}
        fixed_costs::Vector{Float64}
        costs::Matrix{Float64}
    end

    data = ModelKeywordData(
        2,
        1,
        [2.0, 2.0],
        [1.0],
        [1.0, 1.0],
        reshape([1.0, 2.0], 2, 1),
    )
    @test supertype(ModelKeywordData) === Any

    function keyword_master_model!(model::Model, data::ModelKeywordData)
        @variable(model, x[1:data.n_facilities], Bin)
        @variable(model, t >= 0)
        @objective(model, Min, sum(data.fixed_costs[i] * x[i] for i in 1:data.n_facilities) + t)
        @constraint(model, sum(data.capacities[i] * x[i] for i in 1:data.n_facilities) >= sum(data.demands))
        return (x = x,), t
    end

    function keyword_subproblem_model!(model::Model, data::ModelKeywordData, scen_idx::Int; x)
        @variable(model, y[1:data.n_facilities, 1:data.n_customers] >= 0)
        @objective(model, Min, sum(data.costs[i, j] * y[i, j] for i in 1:data.n_facilities, j in 1:data.n_customers))
        @constraint(model, demand[j in 1:data.n_customers], sum(y[:, j]) == data.demands[j])
        @constraint(model, facility_open[i in 1:data.n_facilities, j in 1:data.n_customers], y[i, j] <= x[i])
        return nothing
    end

    optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)

    master = Master(data; model = keyword_master_model!, optimizer = optimizer)
    @test master.dim_x == data.n_facilities
    @test master.dim_t == 1

    classical = ClassicalOracle(data, master; model = keyword_subproblem_model!, optimizer = optimizer)
    unified = UnifiedOracle(data, master; model = keyword_subproblem_model!, optimizer = optimizer)
    pareto = ParetoOracle(data, master, ParetoOracleParam(fill(0.5, data.n_facilities)); model = keyword_subproblem_model!, optimizer = optimizer)
    separable = SeparableOracle(data, master, ClassicalOracle, 1; model = keyword_subproblem_model!, optimizer = optimizer)
    explicit_separable = SeparableOracle(master, [classical])
    knapsack = CFLKnapsackOracle(data, master; model = keyword_subproblem_model!, optimizer = optimizer)

    preprocessing = NoPreprocessing()
    lazy_callback = LazyCallback(classical)
    user_callback = UserCallback(unified)
    bnb_param = BendersBnBParam(time_limit = 10.0, verbose = false)
    explicit_bnb = BendersBnB(
        master;
        lazy_callback = lazy_callback,
        preprocessing = preprocessing,
        user_callback = user_callback,
        param = bnb_param,
    )
    convenience_bnb = BendersBnB(
        master,
        classical;
        preprocessing = preprocessing,
        param = bnb_param,
    )

    @test classical.model isa Model
    @test unified.model isa Model
    @test pareto.model isa Model
    @test length(separable.oracles) == 1
    @test explicit_separable.oracles == [classical]
    @test knapsack.model isa Model

    @test explicit_bnb.master === master
    @test explicit_bnb.lazy_callback === lazy_callback
    @test explicit_bnb.preprocessing === preprocessing
    @test explicit_bnb.user_callback === user_callback
    @test explicit_bnb.param === bnb_param

    @test convenience_bnb.master === master
    @test convenience_bnb.lazy_callback.oracle === classical
    @test convenience_bnb.preprocessing === preprocessing
    @test convenience_bnb.user_callback isa NoUserCallback
    @test convenience_bnb.param === bnb_param

end

@testset "SeparableOracle works with explicit non-GLPK optimizer" begin
    struct SeparableData
        n_scenarios::Int
    end
    data = SeparableData(2)

    function update_master_model!(model::Model, data::SeparableData)
        @variable(model, x[1:2], Bin)
        @variable(model, t[1:data.n_scenarios] >= 0)
        @objective(model, Min, sum(t))
        return (x = x,), t
    end

    function update_sub_model!(model::Model, data::SeparableData, scen_idx::Int; x)
        @variable(model, y >= 0)
        @objective(model, Min, y)
        @constraint(model, y >= 1 - x[scen_idx])
        return nothing
    end

    optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
    master = Master(data; model = update_master_model!, optimizer = optimizer)
    oracle = SeparableOracle(data, master, ClassicalOracle, data.n_scenarios; model = update_sub_model!, optimizer = optimizer)
    is_in_L, hyperplanes, f_x = BendersX.generate_cuts(oracle, [0.0, 0.0], [0.0, 0.0])

    @test !is_in_L
    @test length(hyperplanes) == data.n_scenarios
    @test f_x == [1.0, 1.0]
    @test all(occursin("HiGHS", solver_name(suboracle.model)) for suboracle in oracle.oracles)

    oracle.oracles = BendersX.AbstractTypicalOracle[
        SeparableInterruptionTestOracle(BendersX.TimeLimitException("test timeout")),
        SeparableInterruptionTestOracle(nothing),
    ]
    timeout_error = try
        BendersX.generate_cuts(oracle, [0.0, 0.0], [0.0, 0.0])
        nothing
    catch err
        err
    end
    @test timeout_error isa BendersX.TimeLimitException
    @test timeout_error.msg == "test timeout"

    oracle.oracles = BendersX.AbstractTypicalOracle[
        SeparableInterruptionTestOracle(nothing),
        SeparableInterruptionTestOracle(BendersX.UnexpectedModelStatusException("test status")),
    ]
    status_error = try
        BendersX.generate_cuts(oracle, [0.0, 0.0], [0.0, 0.0])
        nothing
    catch err
        err
    end
    @test status_error isa BendersX.UnexpectedModelStatusException
    @test status_error.msg == "test status"

    oracle.oracles = BendersX.AbstractTypicalOracle[
        SeparableInterruptionTestOracle(ArgumentError("test generic error")),
        SeparableInterruptionTestOracle(nothing),
    ]
    generic_error = try
        BendersX.generate_cuts(oracle, [0.0, 0.0], [0.0, 0.0])
        nothing
    catch err
        err
    end
    @test generic_error isa ArgumentError
    @test generic_error.msg == "test generic error"
end

@testset "transfer_scaled_linear_rows_and_bounds_with_types!" begin
    function transferred_status(point::Vector{Float64}; omega0_value::Float64 = 1.0)
        master = Model()
        @variable(master, x[1:2] >= 0)
        @constraint(master, x[1] + x[2] >= 1.0)
        @constraint(master, x[1] + x[2] <= 1.5)
        @constraint(master, x[1] - x[2] == 0.0)

        dcglp = Model(HiGHS.Optimizer)
        @variable(dcglp, omega0)
        @variable(dcglp, omega[1:2])
        @objective(dcglp, Min, 0.0)

        fix(omega0, omega0_value; force = true)
        fix.(omega, point; force = true)

        transfer_scaled_linear_rows_and_bounds_with_types!(master, x, dcglp, omega, omega0)
        optimize!(dcglp)

        return termination_status(dcglp)
    end

    @test transferred_status([0.75, 0.75]) == OPTIMAL
    @test transferred_status([0.2, 0.2]) == INFEASIBLE
    @test transferred_status([0.2, 0.2]; omega0_value = 0.4) == OPTIMAL
    @test transferred_status([1.0, 1.0]) == INFEASIBLE
end

@testset "BendersX model-update functions" begin
    struct EmptyData end
    data = EmptyData()

    @testset "no model-update functions provided (should throw)" begin
        # Master without a model-update function must throw
        master_error = try
            Master(data)
            nothing
        catch e
            e
        end
        @test master_error isa UnimplementedInterfaceException
        @test occursin("BendersX does not know how to formulate a master problem", master_error.msg)
        @test occursin("Define `update_master_model!", master_error.msg)
        @test occursin("Master(data; model = your_builder!)", master_error.msg)

        # Model-based oracle without a subproblem model-update function must throw
        function update_master_model!(model::Model, data::EmptyData)

            @variable(model, u[1:10], Bin)
            @variable(model, t >= -1e6)
            @constraint(model, sum(u) >= 2)
            @objective(model, Min, 1.0 * t)

            return (u = u, ), t
        end
        master = Master(data; model = update_master_model!)
        subproblem_error = try
            ClassicalOracle(data, master)
            nothing
        catch e
            e
        end
        @test subproblem_error isa UnimplementedInterfaceException
        @test occursin("BendersX does not know how to formulate a subproblem", subproblem_error.msg)
        @test occursin("Define `update_sub_model!", subproblem_error.msg)
        @test occursin("Oracle(...; model = your_builder!)", subproblem_error.msg)
    end

    @testset "master variable container Vector{VariableRef}" begin
        function update_master_model!(model::Model, data::EmptyData)

            @variable(model, u[1:10], Bin)
            @variable(model, t >= -1e6)
            @constraint(model, sum(u) >= 2)
            @objective(model, Min, 1.0 * t)

            return (u = u, ), t
        end

        function update_sub_model!(model::Model, data::EmptyData, scen_idx::Int; u)

            @variable(model, y[1:10] >= 0)
            @objective(model, Min, sum(y))
            @constraint(model, y .<= u)
            return nothing
        end

        master = Master(data; model = update_master_model!)
        oracle = ClassicalOracle(data, master; model = update_sub_model!)

        print(master.model)
        print(oracle.model)
    end
    @testset "master variable container Matrix{VariableRef}" begin
        function update_master_model!(model::Model, data::EmptyData)

            @variable(model, u[1:10,1:3], Bin)
            @variable(model, t >= -1e6)
            @constraint(model, sum(u) >= 2)
            @objective(model, Min, 1.0 * t)

            return (u = u, ), t
        end

        function update_sub_model!(model::Model, data::EmptyData, scen_idx::Int; u)

            @variable(model, y[1:10] >= 0)
            @objective(model, Min, sum(y))
            @constraint(model, sum(u) == 1)
            @constraint(model, y .<= u[:,1])
            return nothing
        end

        master = Master(data; model = update_master_model!)
        oracle = ClassicalOracle(data, master; model = update_sub_model!)

        print(master.model)
        print(oracle.model)
    end
    @testset "Multiple master variable containers of varying types" begin
        function update_master_model!(model::Model, data::EmptyData)

            @variable(model, u[1:10], Bin) # Array
            @variable(model, v[3:10, [:A, :B]], Bin) # DenseAxisArray
            @variable(model, w[i=1:3,j=i:10], Bin) # SparseAxisArray
            @variable(model, t >= -1e6)
            @constraint(model, sum(u) >= 2)
            @objective(model, Min, 1.0 * t)

            return (u = u, v = v, w = w), t
        end

        function update_sub_model!(model::Model, data::EmptyData, scen_idx::Int; u, v, w)

            @variable(model, y[1:10] >= 0)
            @objective(model, Min, sum(y))
            @constraint(model, y .<= u)
            @constraint(model, sum(v) == 1)
            @constraint(model, v[10,:A] == 1)
            @constraint(model, w[3,10] <= 1)
            return nothing
        end

        master = Master(data; model = update_master_model!)
        oracle = ClassicalOracle(data, master; model = update_sub_model!)

        print(master.model)
        print(oracle.model)
    end

end
