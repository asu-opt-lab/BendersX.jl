
struct FacilityKnapsackInfo
    costs::Matrix{Float64}
    demands::Vector{Float64}
    capacity::Vector{Float64}
end

"""
    CFLKnapsackOracleParam

Alias for [`BasicOracleParam`](@ref) used by [`CFLKnapsackOracle`](@ref).

It controls numerical tolerances used during cut generation, including the
relative and absolute thresholds applied when deciding whether a cut is
violated.

See also: [`CFLKnapsackOracle`](@ref), [`BasicOracleParam`](@ref)
"""
const CFLKnapsackOracleParam = BasicOracleParam

"""
    CFLKnapsackOracle <: AbstractTypicalOracle

Specialized oracle for the capacitated facility location problem.

This oracle solves the LP subproblem once to recover dual demand information
and then computes facility-wise knapsack values to form a strengthened
classical Benders cut for the CFLP structure.

Like [`ClassicalOracle`](@ref), it also supports generalized bound constraints
returned by the user model-update function.
"""
mutable struct CFLKnapsackOracle <: AbstractTypicalOracle
    param::CFLKnapsackOracleParam

    model::Model
    fixed_x_constraints::Vector{ConstraintRef}
    facility_knapsack_info::FacilityKnapsackInfo

    gbc_lhs::Vector{VariableRef}
    gbc_rhs::Vector{Union{VariableRef, AffExpr}}
    gbc_sense::Vector{GBCBoundType}
    
    function CFLKnapsackOracle(data, master::Master;
                            model = update_sub_model!,
                            scen_idx::Int=-1, 
                            param::CFLKnapsackOracleParam = CFLKnapsackOracleParam(),
                            optimizer = DEFAULT_OPTIMIZER)
        @debug "Building knapsack oracle for CFLP"
        sub_model = Model()
        set_optimizer_checked!(sub_model, optimizer, "CFLKnapsackOracle subproblem model")

        # Copy the master's coupling variables into the submodel (with identical axes and symbols)
        x_copy = copy_variables!(sub_model, master.x_tuple)

        # Collect all copied master variables and add linking constraint
        x = var_from_tuple(x_copy)
        @constraint(sub_model, fix_x, x .== 0)

        # Build the submodel using user-defined model update, passing the copied variables
        result = model(sub_model, data, scen_idx; x_copy...)
        
        # Parse the result to extract GBC information using shared helper
        gbc_lhs, gbc_rhs, gbc_sense = _parse_gbc_result(result, x)

        facility_knapsack_info = scen_idx == -1 ? FacilityKnapsackInfo(data.costs, data.demands, data.capacities) : FacilityKnapsackInfo(data.costs, data.demands[scen_idx], data.capacities)

        new(param, sub_model, fix_x, facility_knapsack_info, gbc_lhs, gbc_rhs, gbc_sense)
    end
    
    CFLKnapsackOracle() = new()
end

function generate_cuts(oracle::CFLKnapsackOracle, x_value::Vector{Float64}, t_value::Vector{Float64}; tol_normalize = 1.0, time_limit = 3600)
    set_time_limit_sec(oracle.model, time_limit)
    
    # Set GBC bounds based on expression evaluation
    _set_gbc_bounds!(oracle.gbc_lhs, oracle.gbc_rhs, oracle.gbc_sense, x_value)
    
    set_normalized_rhs.(oracle.fixed_x_constraints, x_value)
    optimize!(oracle.model)
    if termination_status(oracle.model) == TIME_LIMIT
        throw(TimeLimitException("CFLKnapsackOracle: Time limit reached during cut generation"))
    end

    status = dual_status(oracle.model)

    if status == FEASIBLE_POINT
        sub_obj_val = objective_value(oracle.model)

        μ = dual.(oracle.model[:demand])
        a_t = [-1.0] 
        
        # Get facility knapsack info
        costs = oracle.facility_knapsack_info.costs
        demands = oracle.facility_knapsack_info.demands
        capacity = oracle.facility_knapsack_info.capacity

        # Calculate KP values for each facility
        KP_values = Vector{Float64}(undef, length(capacity))
        for i in 1:length(capacity)
            KP_values[i] = calculate_KP_value(costs[i,:], demands, capacity[i], μ)
        end

        a_x = KP_values # Vector{Float64}
        a_0 = sum(μ) 
        if sub_obj_val >= t_value[1] * (1 + oracle.param.rtol)
            return false, [Hyperplane(a_x, a_t, a_0)], [sub_obj_val]
        else
            return true, [Hyperplane(a_x, a_t, a_0)], deepcopy(t_value)
        end
        
    elseif status == INFEASIBILITY_CERTIFICATE
        if has_duals(oracle.model)
            dual_sub_obj_val = dual_objective_value(oracle.model)
            @info "dual_sub_obj_val = $dual_sub_obj_val"
            
            a_x = dual.(oracle.fixed_x_constraints)
            
            # Accumulate GBC dual values
            _accumulate_gbc_duals!(a_x, oracle.gbc_lhs, oracle.gbc_rhs, oracle.gbc_sense)
            
            a_t = [0.0]
            a_0 = dual_sub_obj_val - a_x' * x_value 
            if dual_sub_obj_val >= oracle.param.zero_tol/tol_normalize
                return false, [Hyperplane(a_x, a_t, a_0)], [Inf]
            else
                return true, [Hyperplane(a_x, a_t, a_0)], [Inf]
            end
        end
        
    else
        throw(UnexpectedModelStatusException("CFLKnapsackOracle: $(status)"))
    end
end

function calculate_KP_value(costs::Vector{Float64}, demands::Vector{Float64}, capacity::Float64, μ::Vector{Float64})
    n = length(demands)
    
    # ratios = Vector{Tuple{Int,Float64}}(undef, n)
    ratios = [(i, (costs[i] * demands[i] - μ[i]) / demands[i]) for i in 1:n if (costs[i] * demands[i] - μ[i]) < 0]
    
    sort!(ratios, by=x->x[2])
    
    kp_value = 0.0
    remaining_capacity = capacity
    z = zeros(n)

    for (i, _) in ratios
        if remaining_capacity >= demands[i]
            kp_value += costs[i] * demands[i] - μ[i]
            remaining_capacity -= demands[i]
            z[i] = 1.0
        else
            fraction = remaining_capacity / demands[i]
            kp_value += (costs[i]*demands[i] - μ[i]) * fraction
            z[i] = fraction
            break
        end
    end

    return kp_value
end
