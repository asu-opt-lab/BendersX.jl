# ============================================================================
# Benders master problems
# ============================================================================

# -----------------------------------------------------------------------------
# Common interface
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

The returned order defines the correspondence with candidate linking-value vectors passed to oracles and with the `a_x` coefficients of [`Hyperplane`](@ref). The variables must belong to the model returned by `master_model(master)`.
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

The returned order defines the correspondence with oracle objective-value vectors, candidate auxiliary-value vectors, and the `a_t` coefficients of [`Hyperplane`](@ref). The variables must belong to the model returned by `master_model(master)`.
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
    evaluate_primal_objective(master::AbstractMaster, linking_vars, auxiliary_vars)

Evaluate the original problem objective at feasible values `linking_vars` for the linking variables and their true auxiliary values `auxiliary_vars`.

Sequential environments use this operation to update the primal bound. It is a behavioral interface so custom master implementations are not required to store objective coefficients in fields equivalent to `Master.c_x` and `Master.c_t`.
"""
function evaluate_primal_objective(
    master::AbstractMaster,
    linking_vars::AbstractVector{<:Real},
    auxiliary_vars::AbstractVector{<:Real},
)
    throw(UnimplementedInterfaceException(
        "AbstractMaster subtype $(typeof(master)) must implement " *
        "`BendersX.evaluate_primal_objective(master::$(typeof(master)), " *
        "linking_vars, auxiliary_vars)`.",
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

# -----------------------------------------------------------------------------
# Concrete master implementations
# -----------------------------------------------------------------------------

include("master.jl")
