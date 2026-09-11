using Test
using BendersX

@testset "BendersX.jl" begin
    include("test_public_api.jl")
    include("test_solver_extensions.jl")
    include("test_callback_metadata.jl")
    include("test_preprocessing.jl")
    include("test_disjunctive_norm_oracles.jl")
    include("test_separable_oracle.jl")
    include("test_suflp.jl")
    include("test_scalarize_constraints.jl")
    include("test_validate_LP.jl")
    include("test_interface.jl")
    include("test_unified_oracle.jl")
    include("test_pareto_oracle.jl")
end
