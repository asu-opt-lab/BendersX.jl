
"""
    ClassicalOracleParam

Alias for [`BasicOracleParam`](@ref) used by [`ClassicalOracle`](@ref).

It controls the numerical tolerances used to determine whether a constraint violation is detected and whether a value is treated as zero.
"""
const ClassicalOracleParam = BasicOracleParam

"""
    ClassicalOracle <: AbstractTypicalOracle

Classical Benders oracle.

`ClassicalOracle` constructs an LP subproblem by copying the master's coupling variables into a subproblem model and fixing their values to the current candidate solution. Dual information from the resulting subproblem is used to generate classical Benders optimality or feasibility cuts.

The user supplies the subproblem through a model-update function. To use generalized bound constraints (GBCs), the model-update function must return GBC information. See [`update_sub_model!`](@ref) for details.

# Fields

- `param::ClassicalOracleParam`: Numerical tolerances controlling cut generation.
- `model::Model`: LP subproblem model used for oracle evaluation.
- `fixed_x_constraints::Vector{ConstraintRef}`: Constraints fixing the copied master variables to the current candidate solution.
- `gbc_lhs::Vector{VariableRef}`: Left-hand sides of the generalized bound constraints.
- `gbc_rhs::Vector{Union{VariableRef, AffExpr}}`: Right-hand sides of the generalized bound constraints.
- `gbc_sense::Vector{GBCBoundType}`: Senses of the generalized bound constraints.

# Constructor

    ClassicalOracle(
        data,
        master::Master;
        model = update_sub_model!,
        scen_idx::Int = 0,
        param::ClassicalOracleParam = ClassicalOracleParam(),
        optimizer = DEFAULT_OPTIMIZER,
    )

Construct a classical Benders oracle from `data` and `master`.

The `model` function formulates the subproblem using the copied master variables. The subproblem must be LP-compatible.

See also: [`ClassicalOracleParam`](@ref), [`UnifiedOracle`](@ref), [`ParetoOracle`](@ref), [`update_sub_model!`](@ref)
"""
mutable struct ClassicalOracle <: AbstractTypicalOracle
    
    param::ClassicalOracleParam

    model::Model
    fixed_x_constraints::Vector{ConstraintRef}

    gbc_lhs::Vector{VariableRef}
    gbc_rhs::Vector{Union{VariableRef, AffExpr}}
    gbc_sense::Vector{GBCBoundType}


    function ClassicalOracle(data, master::Master;
                            model = update_sub_model!,
                            scen_idx::Int=0, 
                            param::ClassicalOracleParam = ClassicalOracleParam(),
                            optimizer = DEFAULT_OPTIMIZER)
    
            @debug "Building classical oracle"
            sub_model = Model()
            set_optimizer_checked!(sub_model, optimizer, "ClassicalOracle subproblem model")

            # Copy the master's coupling variables into the submodel (with identical axes and symbols)
            x_copy = copy_variables!(sub_model, master.x_tuple)

            # Collect all copied master variables and add linking constraint
            x = var_from_tuple(x_copy)
            @constraint(sub_model, fix_x, x .== 0)

            # Build the submodel using user-defined model update, passing the copied variables
            result = model(sub_model, data, scen_idx; x_copy...)
            
            # Validate that the subproblem is LP-compatible for typical oracles
            _validate_lp_compatibility(sub_model)
            
            # Parse the result to extract GBC information
            gbc_lhs, gbc_rhs, gbc_sense = _parse_gbc_result(result, x)

            new(param, sub_model, fix_x, gbc_lhs, gbc_rhs, gbc_sense)
    end

    ClassicalOracle() = new()
end

"""
    generate_cuts(
        oracle::ClassicalOracle,
        x_value::Vector{Float64},
        t_value::Vector{Float64};
        tol_normalize = 1.0,
        time_limit = 3600,
    )

Generate a classical Benders cut for the candidate master solution.

The subproblem is solved with the copied master variables fixed to `x_value`. For a finite optimal subproblem, dual information is used to generate an optimality cut. For an infeasible subproblem with a valid infeasibility certificate, the dual certificate is used to generate a feasibility cut.

The method returns whether the candidate satisfies the oracle's separation criterion, the generated hyperplanes, and the corresponding subproblem objective values.
"""
function generate_cuts(oracle::ClassicalOracle, x_value::Vector{Float64}, t_value::Vector{Float64}; tol_normalize = 1.0, time_limit = 3600)
    set_time_limit_sec(oracle.model, time_limit)
    set_normalized_rhs.(oracle.fixed_x_constraints, x_value)
    
    # Set GBC bounds based on expression evaluation
    _set_gbc_bounds!(oracle.gbc_lhs, oracle.gbc_rhs, oracle.gbc_sense, x_value)
    
    optimize!(oracle.model)
    if termination_status(oracle.model) == TIME_LIMIT
        throw(TimeLimitException("ClassicalOracle: Time limit reached during cut generation"))
    end
    
    status = dual_status(oracle.model)
    if status == FEASIBLE_POINT
        sub_obj_val = objective_value(oracle.model)

        a_x = dual.(oracle.fixed_x_constraints)
        
        # Accumulate GBC dual values
        _accumulate_gbc_duals!(a_x, oracle.gbc_lhs, oracle.gbc_rhs, oracle.gbc_sense)
        
        a_t = [-1.0] 
        a_0 = sub_obj_val - a_x'*x_value 
        if sub_obj_val >= t_value[1] * (1 + oracle.param.rtol) + oracle.param.atol / tol_normalize
            return false, [Hyperplane(a_x, a_t, a_0)], [sub_obj_val]
        else
            return true, [Hyperplane(a_x, a_t, a_0)], [sub_obj_val]
        end

    elseif status == INFEASIBILITY_CERTIFICATE
        if has_duals(oracle.model)
            dual_sub_obj_val = dual_objective_value(oracle.model)
            @debug "dual_sub_obj_val = $dual_sub_obj_val"
            a_x = dual.(oracle.fixed_x_constraints)
            
            # Accumulate GBC dual values
            _accumulate_gbc_duals!(a_x, oracle.gbc_lhs, oracle.gbc_rhs, oracle.gbc_sense)
            
            a_t = [0.0]
            a_0 = dual_sub_obj_val - a_x' * x_value 
            if dual_sub_obj_val >= oracle.param.zero_tol / tol_normalize
                return false, [Hyperplane(a_x, a_t, a_0)], [Inf]
            else
                return true, [Hyperplane(a_x, a_t, a_0)], [Inf]
            end
        else
            throw(UnexpectedModelStatusException(
                "ClassicalOracle: infeasibility certificate was returned " *
                "but no dual information is available."
            ))
        end
    else
        throw(UnexpectedModelStatusException("ClassicalOracle: $(status). This is likely a numerical issue. Please try using other oracles, such as unified oracle or pareto oracle."))
    end
end
