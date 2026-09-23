# ============================================================================
# Benders master problems
# ============================================================================

# -----------------------------------------------------------------------------
# Common interface
# -----------------------------------------------------------------------------

"""
    master_model(master::AbstractMaster) -> Model

Return the JuMP model associated with `master`.

The returned model is the JuMP model maintained by the Master and used by the surrounding Benders algorithm.
"""
function master_model(master::AbstractMaster)
    throw(UnimplementedInterfaceException(
        "AbstractMaster subtype $(typeof(master)) must implement " *
        "`master_model(master::$(typeof(master)))`.",
    ))
end

"""
    linking_variables(master::AbstractMaster) -> Vector{VariableRef}

Return the master's linking variables in their global coefficient order.

This order defines the correspondence with candidate linking-variable values passed to oracles and with the `a_x` coefficients of [`Hyperplane`](@ref). The variables must belong to the model returned by [`master_model`](@ref).
"""
function linking_variables(master::AbstractMaster)
    throw(UnimplementedInterfaceException(
        "AbstractMaster subtype $(typeof(master)) must implement " *
        "`linking_variables(master::$(typeof(master)))`.",
    ))
end

"""
    auxiliary_variables(master::AbstractMaster) -> Vector{VariableRef}

Return the master's auxiliary variables in their global coefficient order.

This order defines the correspondence with candidate auxiliary-variable values, oracle objective values, and the `a_t` coefficients of [`Hyperplane`](@ref). The variables must belong to the model returned by [`master_model`](@ref).
"""
function auxiliary_variables(master::AbstractMaster)
    throw(UnimplementedInterfaceException(
        "AbstractMaster subtype $(typeof(master)) must implement " *
        "`auxiliary_variables(master::$(typeof(master)))`.",
    ))
end

"""
    copy_linking_variable_tuple!(
        model::Model,
        master::AbstractMaster,
    ) -> NamedTuple

Copy the master's linking variables into `model` and return them as the `NamedTuple` used by the subproblem modeling interface.

The returned variables preserve the names, axes, and container structure provided by the master modeling function and are passed as keyword arguments to the subproblem modeling function. The returned `NamedTuple` can be flattened with [`var_from_tuple`](@ref) when a vector of the copied linking variables is needed.
"""
function copy_linking_variable_tuple!(model::Model, master::AbstractMaster)
    throw(UnimplementedInterfaceException(
        "AbstractMaster subtype $(typeof(master)) must implement " *
        "`copy_linking_variable_tuple!(model::Model, master::$(typeof(master)))`.",
    ))
end

"""
    evaluate_objective(
        master::AbstractMaster,
        linking_vars,
        auxiliary_vars,
    )

Evaluate the original problem objective at the supplied values of the linking and auxiliary variables.

The supplied vectors must follow the orders returned by [`linking_variables`](@ref) and [`auxiliary_variables`](@ref), respectively.
"""
function evaluate_objective(
    master::AbstractMaster,
    linking_vars::AbstractVector{<:Real},
    auxiliary_vars::AbstractVector{<:Real},
)
    throw(UnimplementedInterfaceException(
        "AbstractMaster subtype $(typeof(master)) must implement " *
        "`evaluate_objective(master::$(typeof(master)), " *
        "linking_vars, auxiliary_vars)`.",
    ))
end

"""
    add_cuts!(master::AbstractMaster, hyperplanes::Vector{Hyperplane})

Add the supplied Benders cuts to `master` and return the resulting constraint references.

Each `AbstractMaster` implementation defines how cuts are incorporated into its master model.
"""
function add_cuts!(master::AbstractMaster, hyperplanes::Vector{Hyperplane})
    throw(UnimplementedInterfaceException(
        "AbstractMaster subtype $(typeof(master)) must implement " *
        "`add_cuts!(master::$(typeof(master)), hyperplanes)`.",
    ))
end

# -----------------------------------------------------------------------------
# Concrete master implementations
# -----------------------------------------------------------------------------

include("master.jl")
