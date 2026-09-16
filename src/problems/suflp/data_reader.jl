using JSON

"""
    SUFLPData

Data container for the two-stage stochastic uncapacitated facility location
problem (SUFLP).

`demands[s][j]` is the demand of customer `j` in scenario `s`, and
`probabilities[s]` is the probability of scenario `s`. Transportation costs
are shared by all scenarios; scenario-specific transportation costs are
obtained by multiplying `costs[i, j]` by `demands[s][j]`.
"""
struct SUFLPData
    n_facilities::Int
    n_customers::Int
    n_scenarios::Int
    demands::Vector{Vector{Float64}}
    probabilities::Vector{Float64}
    fixed_costs::Vector{Float64}
    costs::Matrix{Float64}
end

function SUFLPData(
    n_facilities::Integer,
    n_customers::Integer,
    n_scenarios::Integer,
    demands::AbstractVector,
    fixed_costs::AbstractVector,
    costs::AbstractMatrix;
    probabilities::AbstractVector = fill(1.0 / n_scenarios, n_scenarios),
)
    return SUFLPData(
        Int(n_facilities),
        Int(n_customers),
        Int(n_scenarios),
        [Float64.(demand) for demand in demands],
        Float64.(probabilities),
        Float64.(fixed_costs),
        Matrix{Float64}(costs),
    )
end

"""
    SUFLPData(data::SCFLPData; probabilities)

Construct an uncapacitated stochastic instance from an [`SCFLPData`](@ref)
instance by dropping its capacity data. Scenario probabilities default to the
uniform distribution.
"""
function SUFLPData(
    data::SCFLPData;
    probabilities::AbstractVector = fill(
        1.0 / data.n_scenarios,
        data.n_scenarios,
    ),
)
    return SUFLPData(
        data.n_facilities,
        data.n_customers,
        data.n_scenarios,
        data.demands,
        probabilities,
        data.fixed_costs,
        data.costs,
    )
end

"""
    SUFLPData(data::UFLPData)

Construct a one-scenario stochastic instance from a deterministic
[`UFLPData`](@ref) instance.
"""
function SUFLPData(data::UFLPData)
    return SUFLPData(
        data.n_facilities,
        data.n_customers,
        1,
        [data.demands],
        [1.0],
        data.fixed_costs,
        data.costs,
    )
end

"""
    read_stochastic_uncapacitated_facility_location_problem(
        filename;
        filepath = get_artifact_path("scflp"),
    ) -> SUFLPData

Read an SUFLP instance from JSON. The expected fields are `n_facilities`,
`n_customers`, `n_scenarios`, `demands`, `fixed_costs`, and `costs`.
`probabilities` is optional and defaults to equal scenario probabilities.

The packaged SCFLP JSON instances use the same fields plus `capacities`, so
they can be reused directly as SUFLP instances; the capacity field is ignored.
"""
function read_stochastic_uncapacitated_facility_location_problem(
    filename::AbstractString;
    filepath = get_artifact_path("scflp"),
)
    json_filename = endswith(lowercase(filename), ".json") ? filename : "$filename.json"
    raw = JSON.parse(read(joinpath(filepath, json_filename), String))
    n_scenarios = Int(raw["n_scenarios"])
    probabilities = get(raw, "probabilities", fill(1.0 / n_scenarios, n_scenarios))
    costs = reduce(hcat, raw["costs"])'

    return SUFLPData(
        Int(raw["n_facilities"]),
        Int(raw["n_customers"]),
        n_scenarios,
        [Float64.(demand) for demand in raw["demands"]],
        Float64.(probabilities),
        Float64.(raw["fixed_costs"]),
        Matrix{Float64}(costs),
    )
end
