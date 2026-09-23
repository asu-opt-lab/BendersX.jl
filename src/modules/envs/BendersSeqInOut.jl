
"""
    BendersSeqInOut <: AbstractBendersSeq

Sequential Benders decomposition with in-out stabilization.

`BendersSeqInOut` applies the in-out stabilization strategy to the sequential
Benders cutting-plane method. The method maintains a stabilizing point and
uses a convex combination of the current master solution and the stabilizing
point as the oracle query point. If progress stalls, the method switches to
Kelley's cutting-plane method.

The in-out stabilization strategy follows Fischetti, Ljubić, and Sinnl (2017)
and Ben-Ameur and Neto (2007).

# Fields
- `master::AbstractMaster`: Master problem module
- `oracle::AbstractOracle`: Oracle used for subproblem evaluation and cut generation
- `param::BendersSeqInOutParam`: Parameters controlling the algorithm and its
  stabilization strategy.
- `preprocessing::AbstractPreprocessing`: Configuration for optionally preprocessing the LP relaxation of the master problem before the Benders iterations.
- `obj_value::Float64`: Objective value of the best solution found
- `termination_status::TerminationStatus`: Termination status of the Benders algorithm. See [`TerminationStatus`](@ref) for available statuses.

The stabilization parameters are specified through [`BendersSeqInOutParam`](@ref):

- `α`: Weight used to update the stabilizing point.
- `λ`: Weight used to form the perturbed query point.
- `stabilizing_x`: Initial stabilizing point.

# Constructor

    BendersSeqInOut(
        master::AbstractMaster,
        oracle::AbstractOracle;
        param::BendersSeqInOutParam = BendersSeqInOutParam(),
        preprocessing::AbstractPreprocessing = NoPreprocessing(),
    )

Construct a sequential Benders environment with in-out stabilization.

By default, no preprocessing is applied.


# Example
```julia
master = Master(data; model = update_master_model!)
oracle = ClassicalOracle(data, master; model = update_sub_model!)
param = BendersSeqInOutParam(
    α = 0.8,
    λ = 0.5,
    stabilizing_x = zeros(length(BendersX.linking_variables(master))),
)
env = BendersSeqInOut(master, oracle; param = param)
df = solve!(env)
```

# References
- M. Fischetti, I. Ljubić, and M. Sinnl, "Redesigning Benders decomposition for large-scale facility location," Management Science, 63 (2017), 2146–2162.
- W. Ben-Ameur and J. Neto, "Acceleration of cutting-plane and column generation algorithms: Applications to network design," Networks, 49 (2007), 3–17.

See also: [`BendersSeq`](@ref), [`AbstractPreprocessing`](@ref)
"""
mutable struct BendersSeqInOut <: AbstractBendersSeq
    master::AbstractMaster
    oracle::AbstractOracle

    param::BendersSeqInOutParam 

    preprocessing::AbstractPreprocessing

    # result
    obj_value::Float64
    termination_status::TerminationStatus

    function BendersSeqInOut(
        master::AbstractMaster,
        oracle::AbstractOracle;
        param::BendersSeqInOutParam = BendersSeqInOutParam(),
        preprocessing::AbstractPreprocessing = NoPreprocessing(),
    )

        new(master, oracle, param, preprocessing, Inf, NotSolved())
    end
end
"""
    solve!(env::BendersSeqInOut) -> DataFrame

Execute the sequential Benders decomposition algorithm with in-out stabilization.

The method first applies the configured preprocessing, if any. At each iteration, the method solves the master problem, updates the stabilizing point, evaluates the oracle at a perturbed query point, and adds the resulting Benders cuts to the master problem. If lower-bound improvement stalls for a prescribed number of iterations, the method switches to Kelley's cutting-plane method.

The iteration history is returned as a `DataFrame`.
"""
function solve!(env::BendersSeqInOut)
    log = BendersSeqLog()
    param = env.param
    try    
        # Apply preprocessing
        log.preprocessing_time = preprocess!(env.master, env.preprocessing; time_limit = get_sec_remaining(log, param))
        model = master_model(env.master)
        linking_vars = linking_variables(env.master)
        auxiliary_vars = auxiliary_variables(env.master)

        stabilizing_x = param.stabilizing_x
        α = param.α
        λ = param.λ
        kelley_mode = false
        
        while true
            get_sec_remaining(log, param) <= 0.0 && throw(TimeLimitException("BendersSeqInOut: Time limit reached."))

            state = BendersSeqState()
            state.total_time = @elapsed begin
                # Solve master problem
                state.master_time = @elapsed begin
                    set_time_limit_sec(model, get_sec_remaining(log, param))
                    optimize!(model)
                    if is_solved_and_feasible(model; allow_local = false, dual = false)
                        state.LB = JuMP.objective_value(model)
                        state.values[:x] = JuMP.value.(linking_vars)
                        state.values[:t] = JuMP.value.(auxiliary_vars)
                    elseif termination_status(model) == TIME_LIMIT
                        throw(TimeLimitException("BendersSeqInOut: Time limit reached during master solving"))
                    else
                        throw(UnexpectedModelStatusException("BendersSeqInOut: master $(termination_status(model))"))
                    end
                end
                
                # perturb point
                stabilizing_x = α * stabilizing_x + (1 - α) * state.values[:x]
                intermediate_x = λ * state.values[:x] + (1 - λ) * stabilizing_x

                # Execute oracle
                state.oracle_time = @elapsed begin
                    intermediate_is_in_L, hyperplanes, state.f_x = generate_cuts(env.oracle, intermediate_x, state.values[:t]; time_limit = get_sec_remaining(log, param))

                    # In in-out mode, continue moving the query point toward the candidate
                    # until it leaves L or Kelley mode is reached.
                    if intermediate_is_in_L && !kelley_mode
                        while λ < 1.0
                            λ = min(λ + 0.1, 1.0)
                            intermediate_x = λ * state.values[:x] + (1 - λ) * stabilizing_x
                            param.verbose && println("Current intermediate point is in L, making it closer to the candidate point (λ = $λ).")
    
                            intermediate_is_in_L, hyperplanes, state.f_x = generate_cuts(env.oracle, intermediate_x, state.values[:t]; time_limit = get_sec_remaining(log, param))
    
                            intermediate_is_in_L || break

                            if λ == 1.0
                                kelley_mode = true
                                param.verbose && println("Switching to Kelley's cutting plane method (λ = 1.0)")
                            end
                        end
                    end

                    state.is_in_L = kelley_mode && intermediate_is_in_L

                    if kelley_mode && all(isfinite, state.f_x)
                        # In Kelley mode, intermediate_x is the candidate point.
                        update_upper_bound_and_gap!(
                            state,
                            log,
                            (auxiliary_values, linking_values) ->
                                evaluate_objective(
                                    env.master,
                                    linking_values,
                                    auxiliary_values,
                                ),
                        )
                    end
                end
            
                # Update state and record information
                record_iteration!(log, state)
            end

            param.verbose && print_iteration_info(state, log)

            # Check termination criteria
            is_terminated(state, log, param) && break

            # add generated cuts to master
            add_cuts!(env.master, hyperplanes)
            
            # whether to switch kelley mode due to slow progress
            if !kelley_mode
                check_lb_improvement!(state, log; zero_tol = 1e-8, tol_imprv = 5e-4)

                if log.consecutive_no_improvement >= 5
                    # Reset λ to 1 (switch to Kelley's cutting plane)
                    λ = 1.0
                    kelley_mode = true
                    param.verbose && println("Switching to Kelley's cutting plane method (λ = 1.0)")
                end
            end
        end
        env.termination_status = Optimal()
        env.obj_value = log.iterations[end].UB
        
        return to_dataframe(log)
    catch e
        if e isa TimeLimitException
            @warn e.msg
            env.termination_status = TimeLimit()
            env.obj_value = isempty(log.iterations) ? Inf : log.iterations[end].UB
        elseif e isa UnexpectedModelStatusException
            @warn e.msg
            env.termination_status = InfeasibleOrNumericalIssue()
        else
            rethrow()  
        end
        if env.param.verbose
            println("BendersSeqInOut: Terminated with $(env.termination_status)")
        end
        return to_dataframe(log)
    end
end
