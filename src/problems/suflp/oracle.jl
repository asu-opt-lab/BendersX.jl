"""
    UFLKnapsackOracle(data::SUFLPData; scen_idx, param = UFLKnapsackOracleParam())

Construct a customer-disaggregated UFL knapsack oracle for one SUFLP
scenario. Scenario probability is deliberately not applied here because it is
already represented by the auxiliary-variable coefficients in the master.
"""
function UFLKnapsackOracle(
    data::SUFLPData;
    scen_idx::Int,
    param::UFLKnapsackOracleParam = UFLKnapsackOracleParam(),
)
    scenario_data = UFLPData(
        data.n_facilities,
        data.n_customers,
        data.demands[scen_idx],
        data.fixed_costs,
        data.costs,
    )
    return UFLKnapsackOracle(scenario_data; param = param)
end

"""
    UFLKnapsackOracle(
        data::SUFLPData,
        master::Master;
        scen_idx,
        param = UFLKnapsackOracleParam(),
        ...,
    )

Scenario-aware constructor used by the homogeneous [`SeparableOracle`](@ref)
constructor. `model` and `optimizer` are accepted for interface compatibility;
the closed-form knapsack oracle does not build an optimization model.
"""
function UFLKnapsackOracle(
    data::SUFLPData,
    master::AbstractMaster;
    model = update_sub_model!,
    scen_idx::Int,
    param::UFLKnapsackOracleParam = UFLKnapsackOracleParam(),
    optimizer = DEFAULT_OPTIMIZER,
)
    return UFLKnapsackOracle(data; scen_idx = scen_idx, param = param)
end
