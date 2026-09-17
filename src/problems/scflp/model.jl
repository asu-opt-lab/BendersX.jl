function update_master_model!(model::Model, data::SCFLPData)
    I, N = data.n_facilities, data.n_scenarios
    @variable(model, x[1:I], Bin)
    @variable(model, t[1:N] >= -1e6)

    @objective(model, Min, data.fixed_costs'* x + fill(1/N, N)' * t)

    max_demand = maximum(sum(demands) for demands in data.demands)
    @constraint(model, capacity, sum(data.capacities[i] * x[i] for i in 1:I) >= max_demand)

    return (x = x, ), t
end

function update_sub_model!(model::Model, data::SCFLPData, subproblem_idx::Int; x)
    I, J = data.n_facilities, data.n_customers
    @variable(model, y[1:I, 1:J] >= 0)
    # Set objective
    cost_demands = data.costs .* data.demands[subproblem_idx]'
    @objective(model, Min, sum(cost_demands .* y))
    # Add constraints
    @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
    @constraint(model, facility_open, y .<= x)
    @constraint(model, capacity[i in 1:I], sum(data.demands[subproblem_idx][:] .* y[i,:]) <= data.capacities[i] * x[i])
    return nothing
end
