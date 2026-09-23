
"""
    infeasibility_report(master::AbstractMaster, linking_values, auxiliary_values)

Generate and display an infeasibility report for a candidate solution of `master`.

The candidate values must follow the orders returned by [`linking_variables`](@ref) and [`auxiliary_variables`](@ref). The report shows the primal feasibility of the candidate, its objective value computed by [`evaluate_objective`](@ref), and the objective obtained after fixing the master variables to the candidate and resolving the model.

This diagnostic temporarily fixes the linking and auxiliary variables to the supplied candidate values, resolves the master model, and restores their original fixing state before returning.
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
    @info evaluate_objective(master, linking_values, auxiliary_values)

    variables = [linking_vars; auxiliary_vars]
    was_fixed = is_fixed.(variables)
    fixed_values = [
        was_fixed[i] ? fix_value(variables[i]) : 0.0
        for i in eachindex(variables)
    ]

    try
        for variable in keys(opt_sol)
            fix(variable, opt_sol[variable]; force = true)
        end
        optimize!(model)
        @info objective_value(model)
    finally
        for i in eachindex(variables)
            if was_fixed[i]
                fix(variables[i], fixed_values[i]; force = true)
            else
                unfix(variables[i])
            end
        end
    end

    return nothing
end
