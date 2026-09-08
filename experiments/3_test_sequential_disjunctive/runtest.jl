using BendersX
using Test
using JuMP

# Issues:
# 1. SplitOracle/LpDistanceNormalization: Setting CPX_PARAM_EPRHS tightly (e.g., < 1e-6) results in dcglp terminating with ALMOST_INFEASIBLE --> set it tightly (1e-9) and outputs a typical Benders cut when dcglp is ALMOST_INFEASIBLE
# 2. SplitOracle/LpDistanceNormalization: Setting zero_tol in solve_dcglp! large (e.g., 1e-4) results in disjunctive Benders with reuse_dcglp = true terminating with incorrect solution --> set it tightly as 1e-9
# 3. SplitOracle/LpDistanceNormalization: solve_dcglp! becomes stall since the true violation should be multipled with omega_value[:z], which can be fairly small --> terminate dcglp when LB does not improve for a certain number of iterations
# 4. SplitOracle/LpDistanceNormalization: the fat-knapsack-based disjunctive cut may have a sparse gamma_t, so adding only disjunctive cut does not improve lower bound, add_benders_cuts_to_master should be set at true

@testset "Sequential Disjunctive Tests" begin
    @info "Running Sequential Disjunctive Tests"
    include("ufl.jl")
    include("cfl.jl")
    include("scfl.jl")
    include("scfl_separable_split.jl")
    include("snip.jl")
    @info "Sequential Disjunctive Tests completed"
end
