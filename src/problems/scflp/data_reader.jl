using JSON

"""
    SCFLPData

Data container for the stochastic capacitated facility location problem.
"""
struct SCFLPData
    n_facilities::Int
    n_customers::Int
    n_scenarios::Int
    capacities::Vector{Float64}
    demands::Vector{Vector{Float64}}
    fixed_costs::Vector{Float64}
    costs::Matrix{Float64}
end

"""
    read_stochastic_capacited_facility_location_problem(filename; filepath = get_artifact_path("scflp")) -> SCFLPData

Read a stochastic CFLP instance from the packaged JSON dataset.
"""
function read_stochastic_capacited_facility_location_problem(filename::String; filepath=get_artifact_path("scflp"))
    fullpath = joinpath(filepath, "$(filename).json")
    data = JSON.parse(read(fullpath, String))
    costs = reduce(hcat, data["costs"])'
    return SCFLPData(data["n_facilities"], data["n_customers"], data["n_scenarios"],
                     data["capacities"], data["demands"], data["fixed_costs"], costs)
end
