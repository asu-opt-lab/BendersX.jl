"""
    UFLKnapsackOracleParam <: AbstractOracleParam

Parameter type for [`UFLKnapsackOracle`](@ref).

This type controls algorithmic and numerical aspects of the knapsack-based
separation routine used for the uncapacitated facility location problem.

# Fields
- `slim::Bool`: If `true`, aggregate generated cuts into a single hyperplane.
- `add_only_violated_cuts::Bool`: If `true`, skip non-violated per-customer cuts.
- `rtol::Float64`: Relative tolerance used to decide whether a cut is violated.

# Constructor
```julia
UFLKnapsackOracleParam(; slim = false,
                         add_only_violated_cuts = false,
                         rtol = 1e-9)
```

See also: [`UFLKnapsackOracle`](@ref)
"""
mutable struct UFLKnapsackOracleParam <: AbstractOracleParam
    slim::Bool
    add_only_violated_cuts::Bool
    rtol::Float64

    function UFLKnapsackOracleParam(; slim = false, add_only_violated_cuts = false, rtol = 1e-9)
        new(slim, add_only_violated_cuts, rtol)
    end
end
"""
    UFLKnapsackOracle <: AbstractTypicalOracle

Specialized oracle for the uncapacitated facility location problem.

This oracle exploits the closed-form knapsack structure of the UFLP recourse
function to generate cuts without solving a generic LP subproblem. It is most
useful when the master variables correspond to facility-opening decisions.

See also: [`SeparableOracle`](@ref), [`ClassicalOracle`](@ref)
"""
mutable struct UFLKnapsackOracle <: AbstractTypicalOracle
    
    param::UFLKnapsackOracleParam
    
    sorted_cost_demands::Vector{Vector{Float64}}
    sorted_indices::Vector{Vector{Int}}

    J::Int
    obj_values::Vector{Float64}

    function UFLKnapsackOracle(data::UFLPData; 
        scen_idx::Int=-1, 
        param::UFLKnapsackOracleParam = UFLKnapsackOracleParam())
            @debug "Building knapsack oracle for UFLP"
            
            J = data.n_customers
            cost_demands = [data.costs[:,j] .* data.demands[j] for j in 1:J]
            sorted_indices = [sortperm(cost_demands[j]) for j in 1:J]
            sorted_cost_demands = [cost_demands[j][sorted_indices[j]] for j in 1:J]

            obj_values = Vector{Float64}(undef, J)

            new(param, sorted_cost_demands, sorted_indices, J, obj_values)
    end

    UFLKnapsackOracle() = new()
end

auxiliary_dimension(oracle::UFLKnapsackOracle) = oracle.J

function generate_cuts(oracle::UFLKnapsackOracle, x_value::Vector{Float64}, t_value::Vector{Float64}; tol_normalize = 1.0, time_limit = 3600.0)
    tic = time()
    critical_facility = Vector{Int}(undef, oracle.J)
    for j in 1:oracle.J
        sorted_indices = oracle.sorted_indices[j]
        c_sorted = oracle.sorted_cost_demands[j]
        x_sorted = x_value[sorted_indices]

        # Find critical item and calculate contribution
        k = find_critical_item(c_sorted, x_sorted)

        # Calculate objective value contribution
        oracle.obj_values[j] = c_sorted[k] - (k > 1 ? sum((c_sorted[k] - c_sorted[i]) * x_sorted[i] for i in 1:k-1) : 0)

        if oracle.obj_values[j] >= t_value[j] * (1 + oracle.param.rtol)
            critical_facility[j] = k
        else
            critical_facility[j] = oracle.param.add_only_violated_cuts ? -1 : k
        end
    end
    
    if get_sec_remaining(tic, time_limit) <= 0.0
        throw(TimeLimitException("UFLKnapsackOracle: Time limit reached during cut generation"))
    end

    customers = findall(x -> x != -1, critical_facility)

    # is_in_L should be determined by the sum of t's, must not individually
    is_in_L = sum(oracle.obj_values) >= sum(t_value) * (1 + oracle.param.rtol) ? false : true

    hyperplanes = Vector{Hyperplane}()
    for j in customers
        k = critical_facility[j] 
        sorted_indices = oracle.sorted_indices[j]
        c_sorted = oracle.sorted_cost_demands[j]
        
        h = Hyperplane(length(x_value), oracle.J)
        h.a_t[j] = -1.0
        h.a_0 = c_sorted[k]
        for i=1:k-1
            h.a_x[sorted_indices[i]] = -(c_sorted[k] - c_sorted[i])
        end
        push!(hyperplanes, h)
    end

    if is_in_L
        return !(oracle.param.slim) ? (true, hyperplanes, deepcopy(t_value)) : (true, [aggregate(hyperplanes)], deepcopy(t_value))
    end
    return !(oracle.param.slim) ? (false, hyperplanes, deepcopy(oracle.obj_values)) : (false, [aggregate(hyperplanes)], deepcopy(oracle.obj_values))
end

function find_critical_item(c::Vector{Float64}, x::Vector{Float64})
    
    sum_x::Float64 = 0.0
    for (idx, val) in enumerate(x)
        sum_x += val
        if sum_x >= 1.0
            return idx
        end
    end
    throw(AlgorithmException("UFLKnapsackOracle: `k` cannot be `nothing` as sum(x) >= 2 is enforced. Check the models."))
end
