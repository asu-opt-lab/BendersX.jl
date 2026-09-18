"""
    update_master_model!(model::Model, data::UFLPData)

Formulate the standard UFLP master problem in `model`.

The formulation uses binary facility-opening variables `x` and a single auxiliary variable `t` representing the second-stage value approximation.

Returns `(x = x,)` and `t` for use by the Benders decomposition.
"""
function update_master_model!(model::Model, data::UFLPData)
    @variable(model, x[1:data.n_facilities], Bin)
    @variable(model, t >= -1e6)

    @constraint(model, sum(x) >= 2)

    @objective(model, Min, data.fixed_costs'* x + 1.0 * t)

    return (x = x, ), t
end

"""
    update_sub_model!(model::Model, data::UFLPData, subproblem_idx::Int; x)

Formulate the UFLP subproblem in `model`.

The subproblem assigns each customer to an open facility and minimizes the demand-weighted assignment cost. The master variables `x` link facility availability to the assignment variables.

The argument `subproblem_idx` is unused because the UFLP formulation is deterministic.

Returns `nothing`.
"""
function update_sub_model!(model::Model, data::UFLPData, subproblem_idx::Int; x)
    I, J = data.n_facilities, data.n_customers

    @variable(model, y[1:I, 1:J] >= 0)

    cost_demands = data.costs .* data.demands'
    @objective(model, Min, sum(cost_demands .* y))

    @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)
    @constraint(model, facility_open[j in 1:J], y[:, j] .<= x)
    return nothing
end

"""
    update_sub_gbc_model!(model::Model, data::UFLPData, subproblem_idx::Int; x)

Formulate the UFLP subproblem using generalized bound constraints (GBCs).

The assignment and objective components are identical to those of [`update_sub_model!`](@ref), but the linking constraints `y[i, j] <= x[i]` are returned as GBCs rather than added directly to `model`.

The argument `subproblem_idx` is unused because the UFLP formulation is deterministic.

Returns `(gbc_lhs, gbc_rhs, gbc_sense)`, representing the linking constraints `y[i, j] <= x[i]`.
"""
function update_sub_gbc_model!(model::Model, data::UFLPData, subproblem_idx::Int; x)
    
    I, J = data.n_facilities, data.n_customers
    @variable(model, y[1:I, 1:J] >= 0)

    cost_demands = data.costs .* data.demands'
    @objective(model, Min, sum(cost_demands .* y))

    @constraint(model, demand[j in 1:J], sum(y[:,j]) == 1)

    # Return GBC tuple: y[i,j] <= x[i] for each j
    gbc_lhs = vec(y)
    gbc_rhs = [x[i] for j in 1:J for i in 1:I]  # j outer, i inner to match vec(y)
    gbc_sense = fill(UpperBound, I*J)
    return gbc_lhs, gbc_rhs, gbc_sense
end


"""
    update_knapsack_master_model!(model::Model, data::UFLPData)

Formulate the UFLP master problem for use with the knapsack oracle.

The formulation uses binary facility-opening variables `x` and one auxiliary variable `t[j]` for each customer.

Returns `(x = x,)` and `t` for use by the Benders decomposition.
"""
function update_knapsack_master_model!(model::Model, data::UFLPData)
    
    @variable(model, x[1:data.n_facilities], Bin)
    @variable(model, t[1:data.n_customers] >= -1e6)
    @constraint(model, sum(x) >= 2)
    @objective(model, Min, data.fixed_costs'* x + sum(t))
    return (x = x, ), t
end
