using BendersX
using CSV
using DataFrames
using JuMP
include(normpath(joinpath(@__DIR__, "..", "solver_defaults.jl")))

const OUTPUT_DIR = @__DIR__

function create_mip_model(data::UFLPData)
    model = Model(mip_optimizer)
    I, J = data.n_facilities, data.n_customers
    @variable(model, x[1:I], Bin)
    @variable(model, y[1:I, 1:J] >= 0)
    @variable(model, t[1:J] >= 0)

    cost_demands = data.costs .* data.demands'
    @objective(model, Min, data.fixed_costs' * x + sum(t))

    @constraint(model, obj[j in 1:J], t[j] >= sum(cost_demands[:, j] .* y[:, j]))
    @constraint(model, demand[j in 1:J], sum(y[:, j]) == 1)
    @constraint(model, facility_open, y .<= x)
    return model
end

function create_mip_model(data::CFLPData)
    model = Model(mip_optimizer)
    I, J = data.n_facilities, data.n_customers
    @variable(model, x[1:I], Bin)
    @variable(model, y[1:I, 1:J] >= 0)
    @variable(model, t)

    cost_demands = data.costs .* data.demands'
    @objective(model, Min, data.fixed_costs' * x + t)

    @constraint(model, t >= sum(cost_demands .* y))
    @constraint(model, demand[j in 1:J], sum(y[:, j]) == 1)
    @constraint(model, facility_open, y .<= x)
    @constraint(model, capacity[i in 1:I], sum(data.demands .* y[i, :]) <= data.capacities[i] * x[i])
    @constraint(model, capacity_total, sum(data.capacities[i] * x[i] for i in 1:I) >= sum(data.demands))
    return model
end

function create_mip_model(data::SCFLPData)
    model = Model(mip_optimizer)
    I, J, N = data.n_facilities, data.n_customers, data.n_scenarios

    @variable(model, x[1:I], Bin)
    @variable(model, y[1:I, 1:J, 1:N] >= 0)

    @objective(model, Min,
        (1 / N) * sum(data.costs[i, j] * data.demands[s][j] * y[i, j, s] for i in 1:I, j in 1:J, s in 1:N) +
        data.fixed_costs' * x
    )

    @constraint(model, demand[j in 1:J, s in 1:N], sum(y[:, j, s]) == 1)
    @constraint(model, facility_open[i in 1:I, j in 1:J, s in 1:N], y[i, j, s] <= x[i])
    @constraint(model, capacity[i in 1:I, s in 1:N], sum(data.demands[s][j] * y[i, j, s] for j in 1:J) <= data.capacities[i] * x[i])
    return model
end

function create_mip_model(data::SUFLPData)
    model = Model(mip_optimizer)
    I, J, N = data.n_facilities, data.n_customers, data.n_scenarios

    @variable(model, x[1:I], Bin)
    @variable(model, y[1:I, 1:J, 1:N] >= 0)

    @objective(model, Min,
        sum(data.probabilities[s] * data.costs[i, j] * data.demands[s][j] * y[i, j, s] for i in 1:I, j in 1:J, s in 1:N) +
        data.fixed_costs' * x
    )

    @constraint(model, demand[j in 1:J, s in 1:N], sum(y[:, j, s]) == 1)
    @constraint(model, facility_open[i in 1:I, j in 1:J, s in 1:N], y[i, j, s] <= x[i])
    return model
end

function create_mip_model(data::SNIPData)
    model = Model(mip_optimizer)
    K = data.num_scenarios
    @variable(model, x[1:length(data.D)], Bin)
    @variable(model, y[1:data.num_nodes, 1:K] >= 0)

    @objective(model, Min, sum(data.scenarios[k][3] * y[data.scenarios[k][1], k] for k in 1:K))

    @constraint(model, [k in 1:K], y[data.scenarios[k][2], k] == 1)

    for k in 1:K
        for (idx, (from, to, r, q)) in enumerate(data.D)
            @constraint(model, y[from, k] - q * y[to, k] >= 0)
            @constraint(model, y[from, k] - r * y[to, k] >= -(r - q) * data.ψ[k][to] * x[idx])
        end
        for (from, to, r) in data.A_minus_D
            @constraint(model, y[from, k] - r * y[to, k] >= 0)
        end
    end

    @constraint(model, sum(x) <= data.budget)
    return model
end

function solve_reference_objective(data, instance_name::AbstractString)
    mip_model = create_mip_model(data)
    optimize!(mip_model)
    termination_status(mip_model) == OPTIMAL || error("Reference MILP for $(instance_name) did not terminate optimally: $(termination_status(mip_model))")
    return objective_value(mip_model)
end

function write_reference_csv(filename::AbstractString, rows)
    output_path = joinpath(OUTPUT_DIR, filename)
    df = DataFrame(rows)
    temp_path, temp_io = mktemp(dirname(output_path))
    close(temp_io)
    try
        CSV.write(temp_path, df)
        mv(temp_path, output_path; force = true)
    finally
        isfile(temp_path) && rm(temp_path; force = true)
    end
    @info "Wrote $(nrow(df)) reference objectives to $(output_path)"
end

function build_uflp_rows()
    rows = NamedTuple[]
    for i in setdiff(1:71, [67])
        instance_name = "p$i"
        data = read_uflp_benchmark_data(instance_name)
        push!(rows, (instance_name = instance_name, objective_value = solve_reference_objective(data, instance_name)))
    end
    return rows
end

function build_cflp_rows()
    rows = NamedTuple[]
    for i in setdiff(1:71, [67])
        instance_name = "p$i"
        data = read_cflp_benchmark_data(instance_name)
        push!(rows, (instance_name = instance_name, objective_value = solve_reference_objective(data, instance_name)))
    end
    return rows
end

function build_scflp_rows()
    rows = NamedTuple[]
    for i in 1:5
        instance_name = "f25-c50-s64-r10-$i"
        data = read_stochastic_capacited_facility_location_problem(instance_name)
        push!(rows, (instance_name = instance_name, objective_value = solve_reference_objective(data, instance_name)))
    end
    return rows
end

function build_suflp_rows()
    rows = NamedTuple[]
    for i in 1:5
        instance_name = "f25-c50-s64-r10-$i"
        data = read_stochastic_uncapacitated_facility_location_problem(instance_name)
        push!(rows, (instance_name = instance_name, objective_value = solve_reference_objective(data, instance_name)))
    end
    return rows
end

function build_snip_rows()
    rows = NamedTuple[]
    instance_name = "instance=0;snipno=0;budget=30.0"
    data = read_snip_data(0, 0, 30.0)
    push!(rows, (instance_name = instance_name, objective_value = solve_reference_objective(data, instance_name)))
    return rows
end

function main()
    mkpath(OUTPUT_DIR)
    uflp_rows = build_uflp_rows()
    cflp_rows = build_cflp_rows()
    scflp_rows = build_scflp_rows()
    suflp_rows = build_suflp_rows()
    snip_rows = build_snip_rows()

    write_reference_csv("uflp.csv", uflp_rows)
    write_reference_csv("cflp.csv", cflp_rows)
    write_reference_csv("scflp.csv", scflp_rows)
    write_reference_csv("suflp.csv", suflp_rows)
    write_reference_csv("snip.csv", snip_rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
