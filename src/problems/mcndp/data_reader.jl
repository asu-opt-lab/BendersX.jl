using JSON

struct MCNDPData
    num_nodes::Int
    num_arcs::Int
    num_commodities::Int
    arcs::Vector{Tuple{Int,Int}}
    fixed_costs::Vector{Float64}
    variable_costs::Vector{Float64}
    capacities::Vector{Float64}
    demands::Vector{Tuple{Int,Int,Float64}}
end

function read_mcndp_instance(filename::String; filepath=get_artifact_path("mcndp"))
    fullpath = joinpath(filepath, filename)
    open(fullpath, "r") do f
        readline(f)  # skip filename
        dims = split(readline(f))
        num_nodes = parse(Int, dims[1])
        num_arcs = parse(Int, dims[2])
        num_commodities = parse(Int, dims[3])
        
        arcs = Tuple{Int,Int}[]
        fixed_costs = Float64[]
        variable_costs = Float64[]
        capacities = Float64[]
        
        for _ in 1:num_arcs
            vals = split(readline(f))
            push!(arcs, (parse(Int, vals[1]), parse(Int, vals[2])))
            push!(fixed_costs, parse(Float64, vals[5]))
            push!(variable_costs, parse(Float64, vals[3]))
            push!(capacities, parse(Float64, vals[4]))
        end
        
        demands = Tuple{Int,Int,Float64}[]
        while !eof(f)
            vals = split(readline(f))
            length(vals) >= 3 && push!(demands, (parse(Int, vals[1]), parse(Int, vals[2]), parse(Float64, vals[3])))
        end
        
        return MCNDPData(num_nodes, num_arcs, num_commodities, arcs, fixed_costs, variable_costs, capacities, demands)
    end
end
