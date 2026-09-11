
"""
    ParetoOracleParam <: AbstractOracleParam

Parameters controlling [`ParetoOracle`](@ref).

In addition to the numerical tolerances used for cut generation, `ParetoOracleParam` specifies the core point and its update rule for the Magnanti-Wong procedure.

# Fields

- `rtol::Float64`: Relative tolerance for cut violation detection.
- `atol::Float64`: Absolute tolerance for cut violation detection.
- `zero_tol::Float64`: Threshold below which a value is considered zero.
- `core_point::Vector{Float64}`: Current core point used by the Magnanti-Wong procedure. The core point is updated in place after each oracle evaluation according to `λ`.
- `λ::Float64`: Weight used to update the core point after each oracle evaluation.
- `pareto_tol::Float64`: Absolute tolerance for enforcing the Pareto-optimality constraint.
  Rule of thumb: choose `pareto_tol` about 10x-100x larger than the solver feasibility/optimality tolerances.

  The core point is updated according to `core_point = λ * core_point + (1 - λ) * x_value` where `x_value` is the current candidate solution. A value of `λ = 1.0` means no update (classical behavior), while `λ < 1.0` allows the core point to move towards the current candidate solution, potentially improving cut quality.

# Constructor

    ParetoOracleParam(
        core_point::Vector{Float64};
        rtol = 1e-9,
        atol = 0.0,
        zero_tol = 1e-9,
        λ = 1.0,
        pareto_tol = 1e-7,
    )

The `core_point` must be nonempty and `λ` must belong to [0, 1].
"""
struct ParetoOracleParam <: AbstractOracleParam
    rtol::Float64
    atol::Float64
    zero_tol::Float64
    core_point::Vector{Float64}
    λ::Float64
    pareto_tol::Float64

    function ParetoOracleParam(core_point::Vector{Float64}; rtol = 1e-9, atol = 0.0, zero_tol = 1e-9, λ = 1.0, pareto_tol = 1e-7)
        isempty(core_point) && throw(ArgumentError("ParetoOracleParam: core_point must be provided and non-empty"))
        (λ < 0.0 || λ > 1.0) && throw(ArgumentError("ParetoOracleParam: λ must be in [0, 1], got $λ"))
        new(rtol, atol, zero_tol, core_point, λ, pareto_tol)
    end
end

"""
    ParetoOracle <: AbstractTypicalOracle

Oracle that generates Pareto-optimal cuts using the Magnanti-Wong method proposed by 
Magnanti, T. L., & Wong, R. T. (1981), *Accelerating Benders decomposition: Algorithmic 
enhancement and model selection criteria*, Operations research, 29(3), 464-484.

ParetoOracle uses two models:
1. `model`: Classic subproblem (same as `ClassicalOracle`).
2. `pareto_model`: Primal formulation of Magnanti-Wong problem, obtained by reformulating `model`.

## Magnanti-Wong Problem
```math
\\begin{align*}
\\max \\quad & b^{\\top}\\pi_0 + x_0^{\\top}\\pi_1 \\\\
\\text{s.t.} \\quad & A^{\\top}\\pi_0 + \\pi_1 = 0 \\quad (x) \\\\
& B^{\\top}\\pi_0 \\le d \\quad (y) \\\\
& b^{\\top}\\pi_0 + (x^*)^{\\top}\\pi_1 \\ge \\xi^* - \\epsilon \\quad (\\sigma) \\\\
& \\pi_0 \\geq 0
\\end{align*}
```

where:
- ​\$\\xi^*\$ is the optimal objective value from the classic subproblem.
- ​\$x^*\$ is the current master solution.
- ​\$\\epsilon\$ is `pareto_tol` defined in [`ParetoOracleParam`](@ref).
- ​\$x_0\$ is the core point (dynamically updated if λ < 1.0).
- ​\$\\sigma\$ is the primal variable associated with the Pareto-optimality constraint.

## Primal formulation of Magnanti-Wong Problem
```math
\\begin{align*}
\\min \\quad & d^{\\top}y + (\\xi^*-\\epsilon)\\sigma \\\\
\\text{s.t.} \\quad & Ax + By + b\\sigma \\geq b \\quad (\\pi_0) \\\\
& x + x^*\\sigma = x_0 \\quad (\\pi_1) \\\\
& y \\geq 0, \\sigma \\le 0
\\end{align*}
```

# Constructors
```julia
ParetoOracle(data, master::Master, param::ParetoOracleParam;
             model = update_sub_model!,
             scen_idx::Int = 0)

ParetoOracle(data, master::Master;
            model = update_sub_model!,
            scen_idx::Int = 0,
            param::ParetoOracleParam)
```
The second constructor is for `SeparableOracle`. `ParetoOracleParam` should be provided as `sub_oracle_param` in `SeparableOracle`.

See also [`SeparableOracle`](@ref)

# Arguments
`ParetoOracleParam` is not an alias of `BasicOracleParam`. Users must define own `ParetoOracleParam` and provide it to `ParetoOracle`, as the core point depends on the specific problem being solved. All other arguments are identical to those of [`ClassicalOracle`](@ref).

# Example with a problem-specific param
```julia
param = ParetoOracleParam(fill(sum(data.demands)/sum(data.capacities)+1e-3, data.n_facilities)) # Core point of Capacitated Facility Location Problem
oracle = ParetoOracle(data, master, param)
```

# Fields
- `param::ParetoOracleParam`: [`ParetoOracleParam`](@ref) including necessary tolerances, λ and the core point.
- `model::Model`: Classic subproblem.
- `fixed_x_constraints::Vector{ConstraintRef}`: Linking constraints defined in `model`.
- `pareto_model::Model`: The primal formulation of Magnanti-Wong problem that generates Pareto-optimal cuts.
  The auxiliary variable and linking constraints for this model are stored in
  `pareto_model[:σ]` and `pareto_model[:pareto_fixing_constraints]`.
"""
mutable struct ParetoOracle <: AbstractTypicalOracle

    param::ParetoOracleParam

    # Standard subproblem model
    model::Model
    fixed_x_constraints::Vector{ConstraintRef}

    # Magnanti-Wong pareto model
    pareto_model::Model

    function ParetoOracle(data, master::Master, param::ParetoOracleParam;
                         model = update_sub_model!,
                         scen_idx::Int = 0,
                         optimizer = DEFAULT_OPTIMIZER)

        @debug "Building Pareto oracle"
        sub_model = Model()
        set_optimizer_checked!(sub_model, optimizer, "ParetoOracle standard subproblem model")

        # Copy the master's coupling variables into the submodel (with identical axes and symbols)
        x_copy = copy_variables!(sub_model, master.x_tuple)

        # Collect all copied master variables
        x = var_from_tuple(x_copy)

        # Validate core_point dimension
        dim_x = length(x)
        if length(param.core_point) != dim_x
            throw(DimensionMismatch("ParetoOracle: core_point has length $(length(param.core_point)) but expected $dim_x"))
        end

        # Build the submodel using user-defined model update
        model(sub_model, data, scen_idx; x_copy...)

        # Validate that the subproblem is LP-compatible for typical oracles
        _validate_lp_compatibility(sub_model)

        # ---------------------------------------------------------
        # Build Pareto Model (Re-run construction instead of copying)
        # ---------------------------------------------------------
        pareto_model = Model()
        set_optimizer_checked!(pareto_model, optimizer, "ParetoOracle pareto subproblem model")

        # Copy master variables for pareto model
        pareto_x_copy = copy_variables!(pareto_model, master.x_tuple)
        pareto_x = var_from_tuple(pareto_x_copy)

        # Build pareto model structure
        model(pareto_model, data, scen_idx; pareto_x_copy...)

        # Apply Magnanti-Wong transformations (add σ, etc.)
        _apply_pareto_transformations!(pareto_model, pareto_x)

        # ---------------------------------------------------------
        # Finalize Standard Model
        # ---------------------------------------------------------
        # NOW add fixing constraints to standard model
        @constraint(sub_model, fix_x, x .== 0)

        new(param, sub_model, fix_x, pareto_model)
    end

    ParetoOracle() = new()

    function ParetoOracle(data, master::Master;
                          model = update_sub_model!,
                          scen_idx::Int = 0,
        param::ParetoOracleParam,
        optimizer = DEFAULT_OPTIMIZER)
        return ParetoOracle(data, master, param; model = model, scen_idx = scen_idx, optimizer = optimizer)
    end
end

"""
    _apply_pareto_transformations!(pareto_model::Model, x_vars::Vector{VariableRef})

Reformulate Classic subproblem into Magnanti-Wong primal problem.
"""
function _apply_pareto_transformations!(pareto_model::Model, x_vars::Vector{VariableRef})


    σ = @variable(pareto_model, σ <= 0)

    # `_validate_lp_compatibility` guarantees a LP model 
    # scalarize vectorized linear constraints
    _scalarize_constraints!(pareto_model)

    for con in all_constraints(pareto_model, include_variable_in_set_constraints=false)
        # (>=, <=, ==), and the same coefficient update applies to all of them.
        rhs = normalized_rhs(con)
        # Ax + By (⋚/=) b  -->  Ax + By + b * σ (⋚/=) b
        set_normalized_coefficient(con, σ, rhs)
    end

    @constraint(pareto_model, pareto_fixing_constraints[i in eachindex(x_vars)], x_vars[i] == 0)
end

"""
    generate_cuts(oracle::ParetoOracle, x_value::Vector{Float64}, t_value::Vector{Float64}; 
                  tol_normalize = 1.0, time_limit = 3600)

Generate Pareto-optimal cuts using the Magnanti-Wong method.
"""
function generate_cuts(oracle::ParetoOracle, x_value::Vector{Float64}, t_value::Vector{Float64}; tol_normalize = 1.0, time_limit = 3600)
    λ = oracle.param.λ
    oracle.param.core_point .= λ .* oracle.param.core_point .+ (1 - λ) .* x_value
    t0 = time()

    set_time_limit_sec(oracle.model, time_limit)

    set_normalized_rhs.(oracle.fixed_x_constraints, x_value)

    optimize!(oracle.model)

    if termination_status(oracle.model) == TIME_LIMIT
        throw(TimeLimitException("ParetoOracle: Time limit reached while solving standard model"))
    end

    status = dual_status(oracle.model)

    if status == FEASIBLE_POINT

        sub_obj_val = objective_value(oracle.model)

        if sub_obj_val < t_value[1] * (1 + oracle.param.rtol) + oracle.param.atol / tol_normalize
            return true, [Hyperplane(length(x_value), length(t_value))], [sub_obj_val]
        end

        set_objective_coefficient(oracle.pareto_model, oracle.pareto_model[:σ], sub_obj_val - oracle.param.pareto_tol)

        set_normalized_coefficient.(oracle.pareto_model[:pareto_fixing_constraints], oracle.pareto_model[:σ], x_value)

        set_normalized_rhs.(oracle.pareto_model[:pareto_fixing_constraints], oracle.param.core_point)

        remaining_time = max(time_limit - (time() - t0), 0.0)
        set_time_limit_sec(oracle.pareto_model, remaining_time)

        optimize!(oracle.pareto_model)

        if termination_status(oracle.pareto_model) == TIME_LIMIT
            throw(TimeLimitException("ParetoOracle: Time limit reached while solving pareto model"))
        elseif termination_status(oracle.pareto_model) !== OPTIMAL
            throw(UnexpectedModelStatusException("ParetoOracle: Unexpected termination status $(termination_status(oracle.pareto_model))."))
        end

        pareto_status = dual_status(oracle.pareto_model)
        if pareto_status == FEASIBLE_POINT
            a_x = dual.(oracle.pareto_model[:pareto_fixing_constraints])
            a_t = [-1.0]
            a_0 = sub_obj_val - dot(a_x, x_value)
            if sub_obj_val >= t_value[1] * (1 + oracle.param.rtol) + oracle.param.atol / tol_normalize
                return false, [Hyperplane(a_x, a_t, a_0)], [sub_obj_val]
            else
                return true, [Hyperplane(a_x, a_t, a_0)], [sub_obj_val]
            end
        else
            throw(UnexpectedModelStatusException("ParetoOracle: Unexpected dual status $(pareto_status) for pareto_model. This is likely a numerical issue."))
        end

    elseif status == INFEASIBILITY_CERTIFICATE
        if has_duals(oracle.model)
            dual_sub_obj_val = dual_objective_value(oracle.model)
            @debug "dual_sub_obj_val = $dual_sub_obj_val"

            a_x = dual.(oracle.fixed_x_constraints)
            a_t = [0.0]
            a_0 = dual_sub_obj_val - dot(a_x, x_value)

            if dual_sub_obj_val >= oracle.param.zero_tol / tol_normalize
                return false, [Hyperplane(a_x, a_t, a_0)], [Inf]
            else
                return true, [Hyperplane(a_x, a_t, a_0)], [Inf]
            end
        end
    else
        throw(UnexpectedModelStatusException("ParetoOracle: Unexpected dual status $(status) for standard model. This is likely a numerical issue."))
    end
end
