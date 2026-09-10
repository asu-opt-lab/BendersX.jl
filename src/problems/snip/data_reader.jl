using JSON

"""
    SNIPData

Data container for the stochastic network interdiction problem.
"""
struct SNIPData
    num_nodes::Int
    num_scenarios::Int
    scenarios::Vector{Tuple{Int,Int,Float64}}
    D::Vector{Tuple{Int,Int,Float64,Float64}}
    A_minus_D::Vector{Tuple{Int,Int,Float64}}
    ψ::Vector{Vector{Float64}}
    budget::Float64
end

"""
    read_snip_data(instance_no, snip_no, budget; base_dir = get_artifact_path("snip")) -> SNIPData

Read a stochastic network interdiction instance from the packaged SNIP dataset.

`instance_no` selects the network, `snip_no` selects the interdiction setting,
and `budget` fixes the master budget used in the returned data object.
"""
function read_snip_data(instance_no::Int, snip_no::Int, budget::Float64; base_dir::String=get_artifact_path("snip"))
    # Read sensor installation arcs (D)
    D = Tuple{Int,Int,Float64,Float64}[]
    intd_file = joinpath(base_dir, "intd_arc$(instance_no).txt")
    if isfile(intd_file)
        for line in eachline(intd_file)
            isempty(strip(line)) && continue
            vals = filter(!isempty, split(line, '\t'))
            i, j = parse.(Int, vals[1:2])
            r = parse(Float64, vals[3])
            q = snip_no == 2 ? r * 0.5 : snip_no == 3 ? r * 0.1 : snip_no == 4 ? 0.0 : parse(Float64, vals[4])
            push!(D, (i, j, r, q))
        end
    end

    # Read non-sensor arcs (A_minus_D)
    A_minus_D = Tuple{Int,Int,Float64}[]
    arcgain_file = joinpath(base_dir, "arcgain$(instance_no).txt")
    if isfile(arcgain_file)
        for line in eachline(arcgain_file)
            isempty(strip(line)) && continue
            vals = filter(!isempty, split(line, '\t'))
            push!(A_minus_D, (parse(Int, vals[1]), parse(Int, vals[2]), parse(Float64, vals[3])))
        end
    end

    # Read scenarios
    S = Tuple{Int,Int,Float64}[]
    scenarios_file = joinpath(base_dir, "Scenarios.txt")
    if isfile(scenarios_file)
        for line in eachline(scenarios_file)
            isempty(strip(line)) && continue
            vals = split(strip(line))
            for i in 1:3:length(vals)
                i + 2 <= length(vals) && push!(S, (parse(Int, vals[i]), parse(Int, vals[i+1]), parse(Float64, vals[i+2])))
            end
        end
    end

    # Read psi matrix
    psi_content = replace(read(joinpath(base_dir, "psi.txt"), String), r"\s+" => "")
    arrays = match(r"\[(.*)\]", psi_content).captures[1]
    psi = Vector{Float64}[]
    for arr_str in split(arrays, "],[")
        arr_str = replace(replace(arr_str, "[" => ""), "]" => "")
        row = Float64[]
        for val in split(arr_str, ',')
            val = strip(val)
            !isempty(val) && push!(row, parse(Float64, count(".", val) > 1 ? split(val, '.')[1] * "." * join(split(val, '.')[2:end], "") : val))
        end
        push!(psi, row)
    end

    # Create node mapping
    nodes = Set{Int}()
    for (i, j, _, _) in D; push!(nodes, i, j); end
    for (i, j, _) in A_minus_D; push!(nodes, i, j); end
    for (i, j, _) in S; push!(nodes, i, j); end
    node_mapping = Dict(old => new for (new, old) in enumerate(sort(collect(nodes))))
    
    return SNIPData(
        length(node_mapping), length(S),
        [(node_mapping[i], node_mapping[j], p) for (i, j, p) in S],
        [(node_mapping[i], node_mapping[j], r, q) for (i, j, r, q) in D],
        [(node_mapping[i], node_mapping[j], r) for (i, j, r) in A_minus_D],
        psi, budget
    )
end
