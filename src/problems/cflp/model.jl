"""
    update_master_model!(model::Model, data::CFLPData)

Formulate the CFLP master problem in `model`.

The formulation uses binary facility-opening variables `x` and a single auxiliary variable `t` representing the second-stage value approximation. The master problem also enforces sufficient aggregate facility capacity to satisfy total customer demand.

Returns `(x = x,)` and `t` for use by the Benders decomposition.
"""
function update_master_model!(model::Model, data::CFLPData)
    @variable(model, x[1:data.n_facilities], Bin)
    @variable(model, t >= -1e6)

    @objective(model, Min, data.fixed_costs'* x + t)

    I = data.n_facilities
    @constraint(model, capacity, sum(data.capacities[i] * x[i] for i in 1:I) >= sum(data.demands))

    return (x = x, ), t
end


"""
    update_sub_model!(model::Model, data::CFLPData, scen_idx::Int; x)

Formulate the CFLP subproblem in `model`.

The subproblem assigns customer demand to open facilities while respecting facility capacities and minimizes the demand-weighted assignment cost. The master variables `x` link facility availability and capacity to the assignment variables.

The argument `scen_idx` is unused because the CFLP formulation is deterministic.

Returns `nothing`.
"""
function update_sub_model!(model::Model, data::CFLPData, scen_idx::Int; x)
    I, J = data.n_facilities, data.n_customers
    @variable(model, y[1:I, 1:J] >= 0)
    # Set objective
    cost_demands = data.costs .* data.demands'
    @objective(model, Min, sum(cost_demands .* y))
    # Add constraints
    @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
    @constraint(model, facility_open, y .<= x)
    @constraint(model, capacity[i in 1:I], sum(data.demands[:] .* y[i,:]) <= data.capacities[i] * x[i])
    return nothing
end

"""
    update_sub_gbc_model!(model::Model, data::CFLPData, scen_idx::Int; x)

Formulate the CFLP subproblem using generalized bound constraints (GBCs).

The demand and capacity constraints are included directly in `model`, while the linking constraints `y[i, j] <= x[i]` are returned as GBCs rather than added explicitly to the subproblem.

The argument `scen_idx` is unused because the CFLP formulation is deterministic.

Returns `(gbc_lhs, gbc_rhs, gbc_sense)`, representing the linking constraints `y[i, j] <= x[i]`.
"""
function update_sub_gbc_model!(model::Model, data::CFLPData, scen_idx::Int; x)
    
    I, J = data.n_facilities, data.n_customers
    @variable(model, y[1:I, 1:J] >= 0)

    cost_demands = data.costs .* data.demands'
    @objective(model, Min, sum(cost_demands .* y))

    @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
    @constraint(model, capacity[i in 1:I], sum(data.demands[:] .* y[i,:]) <= data.capacities[i] * x[i])

    # Return GBC tuple: y[i,j] <= x[i]
    gbc_lhs = vec(y)
    gbc_rhs = [x[i] for j in 1:J for i in 1:I]  # j outer, i inner to match vec(y)
    gbc_sense = fill(UpperBound, I*J)
    return gbc_lhs, gbc_rhs, gbc_sense
end
