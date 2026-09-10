
"""
    UnifiedOracleParam <: AbstractOracleParam

Parameter type for [`UnifiedOracle`](@ref).

# Fields
- `rtol::Float64`: Relative tolerance for cut violation detection (default: 1e-9).
- `atol::Float64`: Absolute tolerance for cut violation detection (default: 0.0).
- `zero_tol::Float64`: Threshold below which a value is considered zero (default: 1e-9).
- `w0::Float64`: Weight for the objective bound constraint in the subproblem of `UnifiedOracle` (default: 1.0).

See also: [`UnifiedOracle`](@ref) for the definition of the objective bound constraint.
"""
struct UnifiedOracleParam <: AbstractOracleParam
    rtol::Float64
    atol::Float64
    zero_tol::Float64
    w0::Float64

    function UnifiedOracleParam(; rtol = 1e-9, atol = 0.0, zero_tol = 1e-9, w0 = 1.0)
        w0 > 0 || throw(ArgumentError("UnifiedOracleParam: w0 must be positive, got $w0"))
        new(rtol, atol, zero_tol, w0)
    end
end

"""
    UnifiedOracle <: AbstractTypicalOracle

Oracle that generates unified Benders cuts using the formulation proposed by 
Fischetti, M., Salvagnin, D., & Zanette, A. (2010), *A note on the selection of Benders’ cuts*, Mathematical Programming, 124(1), 175-182.

## Dual Problem (Fischetti et al., 2010)
```math
\\begin{align*}
\\max \\quad & (b-Ax^*)^{\\top}\\pi - \\pi_0\\eta^* \\\\
\\quad & B^{\\top}\\pi \\leq d \\pi_0 \\quad (y) \\\\
\\quad & \\sum_{i \\in I(A)} \\pi_i + w_0\\pi_0  = 1 \\quad (\\sigma) \\\\
& \\pi_0, \\pi \\geq 0
\\end{align*}
```

where:
- ​\$x^*\$ is the current master solution.
- ​\$A\$ is the coefficient matrix of variable \$x\$ in the primal problem.
- ​\$I(A)\$ is the indices of the nonzero rows of \$A\$.
- ​\$\\sigma\$ is the primal variable associated with the normalization constraint.

## Notes
In the primal problem, some rows of \$A\$ may be null rows. In those constraints, \$\\sigma\$ does not appear; that is, its coefficient is zero.

## Dual Problem considered in [`UnifiedOracle`](@ref)
```math
\\begin{align*}
\\max \\quad & b^{\\top}\\pi - \\pi_0\\eta^* + (x^*)^{\\top}\\gamma_1 - (x^*)^{\\top}\\gamma_2 \\\\
\\text{s.t.} \\quad & A^{\\top}\\pi + \\gamma_1 - \\gamma_2 = 0 \\quad (x) \\\\
\\quad & B^{\\top}\\pi \\leq d \\pi_0 \\quad (y) \\\\
\\quad & \\sum_{i \\in I(A)} \\pi_i + w_0\\pi_0 + \\mathbf{1}^{\\top}\\gamma_1 + \\mathbf{1}^{\\top}\\gamma_2 = 1 \\quad (\\sigma) \\\\
& \\pi_0, \\pi, \\gamma_1, \\gamma_2 \\geq 0
\\end{align*}
```

## Primal Problem considered in [`UnifiedOracle`](@ref)
```math
\\begin{align*}
\\min \\quad & \\sigma \\\\
\\text{s.t.} \\quad & Ax + By + \\sigma \\geq b \\quad (\\pi) \\\\
\\quad & -d^{\\top}y + w_0\\sigma \\geq -\\eta^* \\quad (\\pi_0) \\\\
& x + \\sigma \\geq x^* \\quad (\\gamma_1) \\\\
& -x + \\sigma \\geq -x^* \\quad (\\gamma_2) \\\\
& y \\geq 0
\\end{align*}
```

## Notes
We denote \$-d^{\\top}y + w_0\\sigma \\geq -\\eta^*\$ as an objective bound constraint.

# Constructor
```julia
UnifiedOracle(data, master::Master;
              model = update_sub_model!,
              scen_idx::Int = 0, 
              param::UnifiedOracleParam = UnifiedOracleParam())
```
Classic subproblem is reformulated to the primal problem of `UnifiedOracle` inside the constructor.

# Arguments
A default `UnifiedOracleParam` is used if none is provided; users may pass a custom instance.
Fields match [`BasicOracleParam`](@ref) except for `w0`.

See also: [`BasicOracleParam`](@ref)

# Example with custom \$w_0\$
```julia
param = UnifiedOracleParam(w0 = 2.0)  # Higher weight for objective violation
oracle = UnifiedOracle(data, master; param = param)
```

# Fields
- `param::UnifiedOracleParam`: [`UnifiedOracleParam`](@ref) including necessary tolerances and \$w_0\$.
- `model::Model`: The primal problem of `UnifiedOracle` that generates unified Benders cuts.
- `fixing_lb_constraints::Vector{ConstraintRef}`: Lower-bound linking constraints in `model`.
- `fixing_ub_constraints::Vector{ConstraintRef}`: Upper-bound linking constraints in `model`.
- `objective_constraint::ConstraintRef`: Objective bound constraint in `model`
"""
mutable struct UnifiedOracle <: AbstractTypicalOracle
    
    param::UnifiedOracleParam

    model::Model
    
    # Unified constraint structure
    fixing_lb_constraints::Vector{ConstraintRef}
    fixing_ub_constraints::Vector{ConstraintRef}
    objective_constraint::ConstraintRef

    function UnifiedOracle(data, master::Master;
                          model = update_sub_model!,
                          scen_idx::Int = 0, 
                          param::UnifiedOracleParam = UnifiedOracleParam(),
                          optimizer = DEFAULT_OPTIMIZER)
    
        @debug "Building unified oracle"
        sub_model = Model()
        set_optimizer_checked!(sub_model, optimizer, "UnifiedOracle subproblem model")

        # Copy the master's coupling variables into the submodel (with identical axes and symbols)
        x_copy = copy_variables!(sub_model, master.x_tuple)

        # Collect all copied master variables
        x = var_from_tuple(x_copy)

        # Build the submodel using user-defined model update, passing the copied variables
        model(sub_model, data, scen_idx; x_copy...)

        # Validate that the subproblem is LP-compatible for typical oracles
        _validate_lp_compatibility(sub_model)

        # Create oracle instance
        oracle = new()
        oracle.param = param
        oracle.model = sub_model
        oracle.fixing_lb_constraints = ConstraintRef[]
        oracle.fixing_ub_constraints = ConstraintRef[]
        
        # Transform to unified form (x is passed for decision variable detection)
        _apply_unified_transformations!(oracle, x)
        
        return oracle
    end

    UnifiedOracle() = new()
end

"""
    _apply_unified_transformations!(oracle::UnifiedOracle, fixed_x_vars::Vector{VariableRef})

Reformulate Classic subproblem by replacing original constraints and objective function with auxiliary variable `σ`.
"""
function _apply_unified_transformations!(oracle::UnifiedOracle, fixed_x_vars::Vector{VariableRef})
    model = oracle.model
    w0 = oracle.param.w0
    
    # Step 1: Get original objective function (must be done before constraint changes)
    original_objective = objective_function(model)
    
    # Step 2: Add auxiliary variable
    σ = @variable(model, σ)
    
    # Step 3: Rewrite problem constraints with sigma BEFORE creating new constraints
    # This eliminates the need to track which constraints to skip
    _rewrite_problem_constraints_with_sigma!(model, σ, fixed_x_vars)
    
    # Step 4: Create unified fixing constraints 
    # Original: x == value  -->  x + σ >= value AND x - σ <= value
    for x_var in fixed_x_vars
        lb_con = @constraint(model, x_var + σ >= 0)
        push!(oracle.fixing_lb_constraints, lb_con)
        
        ub_con = @constraint(model, x_var - σ <= 0)
        push!(oracle.fixing_ub_constraints, ub_con)
    end
    
    # Step 5: Convert original objective to constraint with w0 weight
    # For minimization: original_obj <= t  -->  -original_obj + w0*σ >= -t
    oracle.objective_constraint = @constraint(model, -original_objective + w0 * σ >= 0)
    
    # Step 6: Set new objective to minimize σ
    @objective(model, Min, σ)
end

"""
    _rewrite_problem_constraints_with_sigma!(model::Model, σ::VariableRef, fixed_x_vars::Vector{VariableRef})

Rewrite problem constraints to include sigma (`σ`) based on constraint sense.
Only constraints that contain decision variables (fixed_x_vars) are relaxed.
"""
function _rewrite_problem_constraints_with_sigma!(model::Model, σ::VariableRef, fixed_x_vars::Vector{VariableRef})
    # Build set of decision variables for fast lookup
    decision_vars_set = Set{VariableRef}(fixed_x_vars)

    # Scalarize Interval/Zeros/Nonnegatives/Nonpositives into scalar GreaterThan/LessThan/EqualTo
    _scalarize_constraints!(model)

    # Process each constraint (excluding variable bounds)
    for con in all_constraints(model, include_variable_in_set_constraints=false)
        if _contains_decision_vars(con, decision_vars_set)
            _rewrite_single_constraint!(model, con, σ)
        end
    end
end

"""
    _contains_decision_vars(con::ConstraintRef, decision_vars_set::Set{VariableRef}) -> Bool

Check if a constraint contains any of the decision variables.
Returns true if the constraint's function references any variable in the decision set.

Note: Constraint function type validation is performed by _validate_lp_compatibility() 
before this function is called, so we only handle VariableRef and AffExpr.
"""
function _contains_decision_vars(con::ConstraintRef, decision_vars_set::Set{VariableRef})
    func = constraint_object(con).func
    
    if func isa VariableRef
        return func in decision_vars_set
    else  # AffExpr (validated by _validate_lp_compatibility)
        return any(var in decision_vars_set for (var, _) in func.terms)
    end
end

"""
    _rewrite_single_constraint!(model::Model, con::ConstraintRef, σ::VariableRef)

Relax the constraints by adding `σ`.
"""
function _rewrite_single_constraint!(model::Model, con::ConstraintRef, σ::VariableRef)
    
    # Get constraint object
    con_obj = constraint_object(con)
    func = con_obj.func
    set = con_obj.set
    
    if set isa MOI.GreaterThan
        # >= constraint: add +σ coefficient
        # Original: f(x) >= rhs  -->  f(x) + σ >= rhs
        set_normalized_coefficient(con, σ, 1.0)
        
    elseif set isa MOI.LessThan
        # <= constraint: add -σ coefficient
        # Original: f(x) <= rhs  -->  f(x) - σ <= rhs
        set_normalized_coefficient(con, σ, -1.0)
        
    else  # MOI.EqualTo (validated by _validate_lp_compatibility)
        # == constraint: split into >= and <= with σ
        rhs = set.value
        original_name = name(con)
        
        # Create >= constraint: f(x) + σ >= rhs
        lb_con = @constraint(model, func >= rhs)
        set_normalized_coefficient(lb_con, σ, 1.0)
        if !isempty(original_name)
            set_name(lb_con, _insert_suffix(original_name, "_lb"))
        end
        
        # Create <= constraint: f(x) - σ <= rhs
        ub_con = @constraint(model, func <= rhs)
        set_normalized_coefficient(ub_con, σ, -1.0)
        if !isempty(original_name)
            set_name(ub_con, _insert_suffix(original_name, "_ub"))
        end
        
        # Delete original equality constraint
        delete(model, con)
    end
end

"""
    generate_cuts(oracle::UnifiedOracle, x_value::Vector{Float64}, t_value::Vector{Float64}; 
                  tol_normalize = 1.0, time_limit = 3600)

Generate Benders cuts using the reformulated primal problem [`UnifiedOracle`](@ref).
"""
function generate_cuts(oracle::UnifiedOracle, x_value::Vector{Float64}, t_value::Vector{Float64}; tol_normalize = 1.0, time_limit = 3600)
    set_time_limit_sec(oracle.model, time_limit)
    
    set_normalized_rhs(oracle.objective_constraint, -t_value[1])
    set_normalized_rhs.(oracle.fixing_lb_constraints, x_value)
    set_normalized_rhs.(oracle.fixing_ub_constraints, x_value)
    
    optimize!(oracle.model)
    
    if termination_status(oracle.model) == TIME_LIMIT
        throw(TimeLimitException("UnifiedOracle: Time limit reached while solving unified subproblem"))
    elseif termination_status(oracle.model) !== OPTIMAL
        throw(UnexpectedModelStatusException("UnifiedOracle: Unexpected termination status $(termination_status(oracle.model))."))
    end
    
    status = dual_status(oracle.model)
    
    if status == FEASIBLE_POINT

        σ_val = objective_value(oracle.model)

        decision_coeffs = dual.(oracle.fixing_lb_constraints) .+ dual.(oracle.fixing_ub_constraints)
        auxiliary_coeffs = dual(oracle.objective_constraint)
        const_term = σ_val + auxiliary_coeffs * t_value[1] - dot(decision_coeffs, x_value)
        
        a_x = decision_coeffs
        a_t = [-auxiliary_coeffs]
        a_0 = const_term
        
        if abs(σ_val) <= oracle.param.zero_tol
            return true, [Hyperplane(a_x, a_t, a_0)], [t_value[1]]
        end
        return false, [Hyperplane(a_x, a_t, a_0)], [Inf]
    else
        throw(UnexpectedModelStatusException("UnifiedOracle: Unexpected dual status $(status)."))
    end
end
