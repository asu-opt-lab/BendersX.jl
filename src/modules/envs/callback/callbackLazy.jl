
"""
    LazyCallback <: AbstractLazyCallback

A lazy-constraint callback for Benders branch-and-bound.

`LazyCallback` uses an [`AbstractOracle`](@ref) to generate Benders cuts whenever the MIP solver encounters an integer candidate solution.

# Fields
- `oracle::AbstractOracle`: Oracle used to evaluate candidate solutions and generate Benders cuts.
"""
struct LazyCallback <: AbstractLazyCallback
    oracle::AbstractOracle
    
    function LazyCallback(oracle::AbstractOracle)
        new(oracle)
    end
end

"""
    lazy_callback(
        cb_data, 
        master::AbstractMaster,
        log::BendersBnBLog, 
        param::BendersBnBParam,
        callback::LazyCallback
    )

Generate and submit Benders cuts for an integer candidate solution.

The callback extracts the candidate values of the master variables, evaluates the callback's oracle, and submits any generated cuts to the solver as lazy constraints. Callback statistics are recorded in `log`.

# Notes
The callback is applied only when the solver reports an integer callback node. Solver-specific callback metadata, such as node counts and depths, is obtained through [`callback_node_count`](@ref) and [`callback_node_depth`](@ref), respectively.
"""
function lazy_callback(
    cb_data,
    master::AbstractMaster,
    log::BendersBnBLog,
    param::BendersBnBParam,
    callback::LazyCallback,
)
    model = master_model(master)
    linking_vars = linking_variables(master)
    auxiliary_vars = auxiliary_variables(master)
    status = JuMP.callback_node_status(cb_data, model)
    if status == MOI.CALLBACK_NODE_STATUS_INTEGER

        state = BendersBnBState()
        
        node_count = callback_node_count(cb_data, model)
        if !isnothing(node_count)
            state.node = node_count
        end

        state.values[:x] = JuMP.callback_value.(cb_data, linking_vars)
        state.values[:t] = JuMP.callback_value.(cb_data, auxiliary_vars)

        state.oracle_time = @elapsed begin
            state.is_in_L, hyperplanes, state.f_x = generate_cuts(callback.oracle, state.values[:x], state.values[:t]; time_limit = get_sec_remaining(log, param))
            cuts = !state.is_in_L ? hyperplanes_to_expression(
                model,
                hyperplanes,
                linking_vars,
                auxiliary_vars,
            ) : []
            state.num_cuts += length(hyperplanes)
        end

        # Add cuts
        for cut in cuts
            cut_constraint = @build_constraint(0 >= cut)
            MOI.submit(model, MOI.LazyConstraint(cb_data), cut_constraint)
        end
        record_node!(log, state, true)
    end
end
