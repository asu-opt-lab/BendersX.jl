
"""
    UserCallbackParam <: AbstractUserCallbackParam

Parameters controlling the execution of a user-cut callback.

# Fields
- `frequency::Int`: Number of fractional callback nodes between user-cut evaluations. Defaults to `50`.
- `node_count::Int`: Maximum solver node count at which cuts may be generated. A value of `-1` disables this limit.
- `depth::Int`: Maximum solver node depth at which cuts may be generated. A value of `-1` disables this limit.
"""
Base.@kwdef struct UserCallbackParam <: AbstractUserCallbackParam
    frequency::Int = 50
    node_count::Int = -1
    depth::Int = -1
end

"""
    UserCallback <: AbstractUserCallback

User-cut callback for Benders branch-and-bound.

`UserCallback` evaluates the Benders oracle at selected fractional branch-and-bound nodes and submits the resulting cuts as user cuts.

# Fields
- `param::UserCallbackParam`: Parameters controlling when the callback is evaluated.
- `oracle::AbstractOracle`: Oracle used to generate Benders cuts.
"""
struct UserCallback <: AbstractUserCallback
    param::UserCallbackParam
    oracle::AbstractOracle
    
    function UserCallback(oracle::AbstractOracle; param=UserCallbackParam())
        new(param, oracle)
    end
end

"""
    user_callback(cb_data, master::AbstractMaster, log::BendersBnBLog, callback::UserCallback)

Generate and submit Benders user cuts at selected fractional branch-and-bound nodes.

The callback first applies the configured frequency, node-count, and depth filters. When a node is selected, it extracts the current fractional solution, evaluates the oracle, and submits the generated cuts as user cuts.

# Notes
Solver-specific callback metadata, such as node counts and depths, is obtained through [`callback_node_count`](@ref) and [`callback_node_depth`](@ref), respectively. When the active solver does not provide a particular metadata accessor, the corresponding filter is not applied.
"""

function user_callback(
    cb_data,
    master::AbstractMaster,
    log::BendersBnBLog,
    param::BendersBnBParam,
    callback::UserCallback,
)
    model = master_model(master)
    x_variables = linking_variables(master)
    t_variables = auxiliary_variables(master)
    status = JuMP.callback_node_status(cb_data, model)
    
    if status == MOI.CALLBACK_NODE_STATUS_FRACTIONAL
        log.fractional_nodes_since_cut += 1
        
        # Check if we should process this node based on frequency
        if log.fractional_nodes_since_cut >= callback.param.frequency
            log.fractional_nodes_since_cut = 0
            
            node_count = callback.param.node_count == -1 ? nothing :
                callback_node_count(cb_data, model)
            node_depth = callback.param.depth == -1 ? nothing :
                callback_node_depth(cb_data, model)

            process_node =
                (isnothing(node_count) || node_count <= callback.param.node_count) &&
                (isnothing(node_depth) || node_depth <= callback.param.depth)
            
            if process_node
                # Create state and get current variable values
                state = BendersBnBState()
                state.values[:x] = JuMP.callback_value.(cb_data, x_variables)
                state.values[:t] = JuMP.callback_value.(cb_data, t_variables)
                
                # Generate cuts
                state.oracle_time = @elapsed begin
                    state.is_in_L, hyperplanes, state.f_x = generate_cuts(callback.oracle, state.values[:x], state.values[:t]; time_limit = get_sec_remaining(log, param))
                    cuts = !state.is_in_L ? hyperplanes_to_expression(model, hyperplanes, x_variables, t_variables) : []
                    state.num_cuts += length(hyperplanes)
                end

                # Add cuts
                for cut in cuts
                    cut_constraint = @build_constraint(0 >= cut)
                    MOI.submit(model, MOI.UserCut(cb_data), cut_constraint)
                end
                
                # Record node information
                record_node!(log, state, false)
            end
        end
    end
end
