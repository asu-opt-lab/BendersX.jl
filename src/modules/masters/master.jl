"""
    Master <: AbstractMaster

Provided [`AbstractMaster`](@ref) implementation that adds all Benders cuts supplied by oracles to the master problem.

# Fields

- `model::Model`: Underlying JuMP optimization model.
- `x_tuple::NamedTuple`: Structured tuple containing the master variables returned by the master modeling function.
- `x::Vector{VariableRef}`: Flattened vector of linking variables.
- `t::Vector{VariableRef}`: Vector of auxiliary variables.
- `dim_x::Int`: Number of linking variables.
- `dim_t::Int`: Number of auxiliary variables.
- `c_x::Vector{Float64}`: Objective coefficients of the linking variables.
- `c_t::Vector{Float64}`: Objective coefficients of the auxiliary variables.

# Constructor

    Master(
        data;
        model = update_master_model!,
        optimizer = DEFAULT_OPTIMIZER,
    )

Construct a `Master` from `data` using the supplied master modeling function.

# Arguments

- `data`: Problem data used to formulate the master problem.
- `model`: Function that builds the master model and returns the linking and auxiliary variables.
- `optimizer`: JuMP-compatible optimizer constructor.
"""
mutable struct Master <: AbstractMaster
    model::Model
    x_tuple::NamedTuple
    x::Vector{VariableRef}
    t::Vector{VariableRef}

    dim_x::Int
    dim_t::Int
    c_x::Vector{Float64}
    c_t::Vector{Float64}

    function Master(data; model=update_master_model!, optimizer = DEFAULT_OPTIMIZER)

        @debug "Building Master module"

        jump_model = Model()
        set_optimizer_checked!(jump_model, optimizer, "Master model")

        x_tuple, t = model(jump_model, data)
        t = t isa VariableRef ? [t] : t
        x = var_from_tuple(x_tuple)

        dim_x = length(x)
        obj = objective_function(jump_model)
        c_x = [coefficient(obj, x[i]) for i in 1:dim_x]
        dim_t = length(t)
        c_t = [coefficient(obj, t[i]) for i in 1:dim_t]

        new(jump_model, x_tuple, x, t, dim_x, dim_t, c_x, c_t)
    end
end

# The provided Master keeps its existing fields for backward compatibility,
# while framework code accesses them only through the documented interface.
master_model(master::Master) = master.model
linking_variables(master::Master) = master.x
auxiliary_variables(master::Master) = master.t
copy_linking_variable_tuple!(model::Model, master::Master) =
    copy_variables!(model, master.x_tuple)

function evaluate_objective(
    master::Master,
    linking_vars::AbstractVector{<:Real},
    auxiliary_vars::AbstractVector{<:Real},
)
    length(linking_vars) == length(master.c_x) || throw(DimensionMismatch(
        "evaluate_objective: expected $(length(master.c_x)) linking " *
        "values, got $(length(linking_vars)).",
    ))
    length(auxiliary_vars) == length(master.c_t) || throw(DimensionMismatch(
        "evaluate_objective: expected $(length(master.c_t)) auxiliary " *
        "values, got $(length(auxiliary_vars)).",
    ))
    return dot(master.c_x, linking_vars) + dot(master.c_t, auxiliary_vars)
end

function add_cuts!(master::Master, hyperplanes::Vector{Hyperplane})
    model = master_model(master)
    cuts = hyperplanes_to_expression(
        model,
        hyperplanes,
        linking_variables(master),
        auxiliary_variables(master),
    )
    return @constraint(model, 0.0 .>= cuts)
end
