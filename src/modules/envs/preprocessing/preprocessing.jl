"""
    AbstractPreprocessing

Abstract supertype for preprocessing strategies in Benders decomposition.

A preprocessing strategy may modify the master problem and generate initial Benders cuts before the sequential Benders or MIP branch-and-bound search begins.
"""
abstract type AbstractPreprocessing end

"""
    NoPreprocessing <: AbstractPreprocessing

No-op preprocessing strategy.

Use this type when no preprocessing or initial Benders cuts should be generated before Benders decomposition.
"""
struct NoPreprocessing <: AbstractPreprocessing end

"""
    preprocess!(master::AbstractMaster, preprocessing::NoPreprocessing) -> Float64
    
No-op implementation for NoPreprocessing.
# Returns
- `Float64`: 0.0 (no time spent)
"""
function preprocess!(master::AbstractMaster, preprocessing::NoPreprocessing; time_limit::Float64 = 100.0)
    return 0.0
end

include("preprocessingLP.jl")
include("preprocessingDisjunctive.jl")
