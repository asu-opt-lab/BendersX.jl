"""
    Master <: AbstractMaster

Master problem used in Benders decomposition.

`Master` stores the JuMP model, the first-stage decision variables `x`, the auxiliary variables `t`, and the corresponding objective coefficients used by the Benders algorithms.

# Fields

- `model::Model`: Underlying JuMP optimization model.
- `x_tuple::NamedTuple`: Named tuple containing the master variables returned by the master-model builder.
- `x::Vector{VariableRef}`: Flattened vector of master variables that link to the second-stage problems.
- `t::Vector{VariableRef}`: Auxiliary variables associated with the second-stage value functions.
- `dim_x::Int`: Dimension of `x`.
- `dim_t::Int`: Dimension of `t`.
- `c_x::Vector{Float64}`: Objective coefficients of `x`.
- `c_t::Vector{Float64}`: Objective coefficients of `t`.

# Constructor

    Master(
        data;
        model = update_master_model!,
        optimizer = DEFAULT_OPTIMIZER,
    )

Construct a Benders master problem from `data`.

# Arguments

- `data`: Problem data used to formulate the master problem. Any Julia object is accepted.
- `model`: Function that builds the master model.
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

# -----------------------------------------------------------------------------
# AbstractMaster interface
# -----------------------------------------------------------------------------

"""
    master_model(master::AbstractMaster) -> Model

Return the JuMP model owned by `master`.

Built-in BendersX environments currently require an `AbstractMaster` to be JuMP-backed. Custom master implementations should specialize this method when their model is stored under a different field name.
"""
function master_model(master::AbstractMaster)
    throw(UnimplementedInterfaceException(
        "AbstractMaster subtype $(typeof(master)) must implement " *
        "`BendersX.master_model(master::$(typeof(master)))`.",
    ))
end

"""
    linking_variables(master::AbstractMaster) -> Vector{VariableRef}

Return the master's linking variables in their global coefficient order.

The returned order defines the correspondence with candidate vectors passed to oracles and with the `a_x` coefficients of [`Hyperplane`](@ref). The variables must belong to the model returned by `master_model(master)`.
"""
function linking_variables(master::AbstractMaster)
    throw(UnimplementedInterfaceException(
        "AbstractMaster subtype $(typeof(master)) must implement " *
        "`BendersX.linking_variables(master::$(typeof(master)))`.",
    ))
end

"""
    auxiliary_variables(master::AbstractMaster) -> Vector{VariableRef}

Return the master's auxiliary variables in their global coefficient order.

The returned order defines the correspondence with oracle objective-value vectors, candidate `t` vectors, and the `a_t` coefficients of [`Hyperplane`](@ref). The variables must belong to the model returned by `master_model(master)`.
"""
function auxiliary_variables(master::AbstractMaster)
    throw(UnimplementedInterfaceException(
        "AbstractMaster subtype $(typeof(master)) must implement " *
        "`BendersX.auxiliary_variables(master::$(typeof(master)))`.",
    ))
end

"""
    copy_linking_variables!(model::Model, master::AbstractMaster) -> NamedTuple

Create copies of the master's linking variables in `model`, preserving the names, axes, and container structure expected by a subproblem model builder.

Flattening the returned `NamedTuple` with [`var_from_tuple`](@ref) must produce the same variable order as `linking_variables(master)`. This operation-oriented interface lets a custom master choose its own internal representation without exposing a field equivalent to `Master.x_tuple`.
"""
function copy_linking_variables!(model::Model, master::AbstractMaster)
    throw(UnimplementedInterfaceException(
        "AbstractMaster subtype $(typeof(master)) must implement " *
        "`BendersX.copy_linking_variables!(model::Model, master::$(typeof(master)))`.",
    ))
end

"""
    evaluate_primal_objective(master::AbstractMaster, x_value, recourse_value)

Evaluate the original problem objective at a feasible linking-variable value `x_value` and its true recourse values `recourse_value`.

Sequential environments use this operation to update the primal bound. It is a behavioral interface so custom master implementations are not required to store objective coefficients in fields equivalent to `Master.c_x` and `Master.c_t`.
"""
function evaluate_primal_objective(
    master::AbstractMaster,
    x_value::AbstractVector{<:Real},
    recourse_value::AbstractVector{<:Real},
)
    throw(UnimplementedInterfaceException(
        "AbstractMaster subtype $(typeof(master)) must implement " *
        "`BendersX.evaluate_primal_objective(master::$(typeof(master)), " *
        "x_value, recourse_value)`.",
    ))
end

"""
    add_cuts!(master::AbstractMaster, hyperplanes::Vector{Hyperplane})

Add ordinary Benders cuts represented by `hyperplanes` to `master` and return the created constraint references.

The default implementation converts the hyperplanes using the public master interface and adds them to the underlying JuMP model. A custom master may override this operation to record or manage cuts without requiring changes to the sequential environments.
"""
function add_cuts!(master::AbstractMaster, hyperplanes::Vector{Hyperplane})
    model = master_model(master)
    cuts = hyperplanes_to_expression(
        model,
        hyperplanes,
        linking_variables(master),
        auxiliary_variables(master),
    )
    return @constraint(model, 0.0 .>= cuts)
end

# The provided Master keeps its existing fields for backward compatibility,
# while framework code accesses them only through the documented interface.
master_model(master::Master) = master.model
linking_variables(master::Master) = master.x
auxiliary_variables(master::Master) = master.t
copy_linking_variables!(model::Model, master::Master) =
    copy_variables!(model, master.x_tuple)

function evaluate_primal_objective(
    master::Master,
    x_value::AbstractVector{<:Real},
    recourse_value::AbstractVector{<:Real},
)
    length(x_value) == length(master.c_x) || throw(DimensionMismatch(
        "evaluate_primal_objective: expected $(length(master.c_x)) linking " *
        "values, got $(length(x_value)).",
    ))
    length(recourse_value) == length(master.c_t) || throw(DimensionMismatch(
        "evaluate_primal_objective: expected $(length(master.c_t)) recourse " *
        "values, got $(length(recourse_value)).",
    ))
    return dot(master.c_x, x_value) + dot(master.c_t, recourse_value)
end
