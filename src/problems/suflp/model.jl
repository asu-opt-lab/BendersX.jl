"""
    update_master_model!(model::Model, data::SUFLPData)

Formulate the scenario-aggregated SUFLP master problem. It contains one
auxiliary variable per scenario and is suitable for scalar-output subproblem
oracles such as [`ClassicalOracle`](@ref), [`UnifiedOracle`](@ref), and
[`ParetoOracle`](@ref).
"""
function update_master_model!(model::Model, data::SUFLPData)
    I, S = data.n_facilities, data.n_scenarios
    @variable(model, x[1:I], Bin)
    @variable(model, t[1:S] >= -1.0e6)

    @constraint(model, sum(x) >= 1)
    @objective(
        model,
        Min,
        data.fixed_costs' * x + sum(data.probabilities[s] * t[s] for s in 1:S),
    )

    return (x = x,), t
end

"""
    update_knapsack_master_model!(model::Model, data::SUFLPData)

Formulate the customer-disaggregated SUFLP master problem for
[`UFLKnapsackOracle`](@ref). The auxiliary variables are indexed as
`t[customer, scenario]` and returned in column-major order, so every scenario
owns one contiguous block of `data.n_customers` variables.
"""
function update_knapsack_master_model!(model::Model, data::SUFLPData)
    I, J, S = data.n_facilities, data.n_customers, data.n_scenarios
    @variable(model, x[1:I], Bin)
    @variable(model, t[1:J, 1:S] >= -1.0e6)

    @constraint(model, sum(x) >= 1)
    @objective(
        model,
        Min,
        data.fixed_costs' * x +
        sum(data.probabilities[s] * t[j, s] for j in 1:J, s in 1:S),
    )

    return (x = x,), vec(t)
end

"""
    update_sub_model!(model::Model, data::SUFLPData, scen_idx::Int; x)

Formulate one unweighted SUFLP scenario subproblem. Scenario probabilities are
applied only in the master objective; keeping them out of the subproblem
prevents double weighting of recourse values and Benders cuts.
"""
function update_sub_model!(model::Model, data::SUFLPData, scen_idx::Int; x)
    I, J = data.n_facilities, data.n_customers
    @variable(model, y[1:I, 1:J] >= 0)

    cost_demands = data.costs .* data.demands[scen_idx]'
    @objective(model, Min, sum(cost_demands .* y))
    @constraint(model, demand[j in 1:J], sum(y[:, j]) == 1)
    @constraint(model, facility_open[i in 1:I, j in 1:J], y[i, j] <= x[i])
    return nothing
end

"""
    update_sub_gbc_model!(model::Model, data::SUFLPData, scen_idx::Int; x)

Formulate one SUFLP scenario subproblem while returning `y[i, j] <= x[i]` as
generalized bound constraints.
"""
function update_sub_gbc_model!(model::Model, data::SUFLPData, scen_idx::Int; x)
    I, J = data.n_facilities, data.n_customers
    @variable(model, y[1:I, 1:J] >= 0)

    cost_demands = data.costs .* data.demands[scen_idx]'
    @objective(model, Min, sum(cost_demands .* y))
    @constraint(model, demand[j in 1:J], sum(y[:, j]) == 1)

    gbc_lhs = vec(y)
    gbc_rhs = [x[i] for j in 1:J for i in 1:I]
    gbc_sense = fill(UpperBound, I * J)
    return gbc_lhs, gbc_rhs, gbc_sense
end
