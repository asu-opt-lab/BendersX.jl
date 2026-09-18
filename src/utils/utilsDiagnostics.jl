"""
    infeasibility_report(master::AbstractMaster, linking_values, auxiliary_values)

Generate and display an infeasibility/consistency check for a candidate solution
to a master model.

This function takes proposed linking and auxiliary values for the master problem,
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

- `linking_values::AbstractVector{<:Real}`
  Candidate values for the linking variables in the order returned by
  [`linking_variables`](@ref).

- `auxiliary_values::AbstractVector{<:Real}`
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
linking_values = [1.0, 0.5, 2.0]
auxiliary_values = [0.3]

infeasibility_report(master, linking_values, auxiliary_values)
"""
function infeasibility_report(master::AbstractMaster, linking_values, auxiliary_values)
    model = master_model(master)
    linking_vars = linking_variables(master)
    auxiliary_vars = auxiliary_variables(master)

    length(linking_values) == length(linking_vars) || throw(DimensionMismatch(
        "infeasibility_report: expected $(length(linking_vars)) linking values, " *
        "got $(length(linking_values)).",
    ))
    length(auxiliary_values) == length(auxiliary_vars) || throw(DimensionMismatch(
        "infeasibility_report: expected $(length(auxiliary_vars)) auxiliary values, " *
        "got $(length(auxiliary_values)).",
    ))

    opt_sol = Dict{VariableRef, Float64}()
    for i in eachindex(linking_vars)
        opt_sol[linking_vars[i]] = linking_values[i]
    end
    for i in eachindex(auxiliary_vars)
        opt_sol[auxiliary_vars[i]] = auxiliary_values[i]
    end

    @info primal_feasibility_report(model, opt_sol)
    @info evaluate_primal_objective(master, linking_values, auxiliary_values)

    for v in keys(opt_sol)
        fix(v, opt_sol[v]; force=true)
    end
    optimize!(model)
    @info objective_value(model)
end
