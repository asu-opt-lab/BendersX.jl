
"""
    UFLPData

Data container for the uncapacitated facility location problem.
"""
struct UFLPData
    n_facilities::Int
    n_customers::Int
    demands::Vector{Float64}
    fixed_costs::Vector{Float64}
    costs::Matrix{Float64}
end

"""
    read_uflp_benchmark_data(filename; filepath = get_artifact_path("uflp_locssall")) -> UFLPData

Read a benchmark UFLP instance from the packaged LOCSSALL-style dataset.
"""
function read_uflp_benchmark_data(filename::AbstractString; filepath=get_artifact_path("uflp_locssall"))
    fullpath = joinpath(filepath, filename)
    f = open(fullpath)

    vals1 = split(readline(f))
    n_facilities = parse(Int, vals1[1])
    n_customers = parse(Int, vals1[2])

    fixed_costs = zeros(Float64, n_facilities)
    for i in 1:n_facilities
        vals = split(readline(f))
        fixed_costs[i] = parse(Float64, vals[2])
    end

    demands = zeros(Float64, n_customers)
    for i in 1:Int(n_customers/10)
        vals = split(readline(f))
        for j in 1:10
            demands[10*(i-1)+j] = parse(Float64, vals[j])
        end
    end

    costs = zeros(Float64, n_facilities, n_customers)
    line_facility = Int(n_customers/10)
    nline = 0
    fth = 1
    while !eof(f)
        vals = split(readline(f))
        nline += 1
        for j in 1:10
            costs[fth, 10*(nline-1)+j] = parse(Float64, vals[j])
        end
        if nline == line_facility
            fth += 1
            nline = 0
        end
    end
    close(f)
    
    return UFLPData(n_facilities, n_customers, demands, fixed_costs, costs)
end

"""
    read_Simple_data(filename; filepath = get_artifact_path("uflp_allkoerkelghosh")) -> UFLPData

Read a UFLP instance from the packaged "Simple" dataset.
"""
function read_Simple_data(filename::AbstractString; filepath=get_artifact_path("uflp_allkoerkelghosh"))
    fullpath = joinpath(filepath, filename)
    f = open(fullpath)

    readline(f)
    vals1 = split(readline(f))
    n_facilities = parse(Int, vals1[1])
    n_customers = parse(Int, vals1[2])

    fixed_costs = zeros(Int, n_facilities)
    costs = zeros(Int, n_facilities, n_customers)

    fth = 1
    while !eof(f)
        vals = split(readline(f))
        fixed_costs[fth] = parse(Int, vals[2])
        for j in 1:n_customers
            costs[fth, j] = parse(Int, vals[2+j])
        end
        fth += 1
    end
    close(f)

    return UFLPData(n_facilities, n_customers, ones(Int, n_customers), fixed_costs, costs)
end
