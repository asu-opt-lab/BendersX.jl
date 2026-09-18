"""
    infeasibility_report(master::AbstractMaster, x_opt, t_opt)

Generate and display an infeasibility/consistency check for a candidate solution
to a master model.

This function takes a proposed solution `(x_opt, t_opt)` for the master problem,
loads it into the JuMP model, and evaluates:

  * primal feasibility of all constraints,
  * the objective value at the candidate point,
  * the objective value after fixing all variables and re-solving the model.

The procedure is typically used for debugging to verify
whether the candidate solution is feasible for the master problem.

# Arguments
- `master::AbstractMaster`
  A master implementation providing the documented [`AbstractMaster`](@ref)
  interface.

- `x_opt::AbstractVector{<:Real}`
  Candidate values for the x-variables.

- `t_opt::AbstractVector{<:Real}`
  Candidate values for the auxiliary variables in the order returned by
  [`auxiliary_variables`](@ref).

# Behavior
1. Converts the candidate values into a dictionary `opt_sol::Dict{VariableRef,Float64}`.
2. Prints:
   - a primal feasibility report (`primal_feasibility_report`),
   - the objective value returned by [`evaluate_primal_objective`](@ref).
3. Fixes all master variables to the candidate values and re-solves the model.
4. Prints the resulting objective value.

# Returns
Nothing.
The function is used for logging and diagnostic purposes.

# Side Effects
- Mutates the JuMP model by fixing all variables (`fix(...; force=true)`).
- Solves the model once using `optimize!`.
- Produces logging output via `@info`.

# Example
```julia
# Suppose `master` has three linking variables and one auxiliary variable.
x_opt = [1.0, 0.5, 2.0]
t_opt = [0.3]

infeasibility_report(master, x_opt, t_opt)
"""
function infeasibility_report(master::AbstractMaster, x_opt, t_opt)
    model = master_model(master)
    x_variables = linking_variables(master)
    t_variables = auxiliary_variables(master)

    length(x_opt) == length(x_variables) || throw(DimensionMismatch(
        "infeasibility_report: expected $(length(x_variables)) linking values, " *
        "got $(length(x_opt)).",
    ))
    length(t_opt) == length(t_variables) || throw(DimensionMismatch(
        "infeasibility_report: expected $(length(t_variables)) auxiliary values, " *
        "got $(length(t_opt)).",
    ))

    opt_sol = Dict{VariableRef, Float64}()
    for i in eachindex(x_variables)
        opt_sol[x_variables[i]] = x_opt[i]
    end
    for i in eachindex(t_variables)
        opt_sol[t_variables[i]] = t_opt[i]
    end

    @info primal_feasibility_report(model, opt_sol)
    @info evaluate_primal_objective(master, x_opt, t_opt)

    for v in keys(opt_sol)
        fix(v, opt_sol[v]; force=true)
    end
    optimize!(model)
    @info objective_value(model)
end
