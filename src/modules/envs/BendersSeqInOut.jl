
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
param = BendersSeqInOutParam(α = 0.8, λ = 0.5, stabilizing_x = zeros(master.dim_x))
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
                    set_time_limit_sec(env.master.model, get_sec_remaining(log, param))
                    optimize!(env.master.model)
                    if is_solved_and_feasible(env.master.model; allow_local = false, dual = false)
                        state.LB = JuMP.objective_value(env.master.model)
                        state.values[:x] = JuMP.value.(env.master.x)
                        state.values[:t] = JuMP.value.(env.master.t)
                    elseif termination_status(env.master.model) == TIME_LIMIT
                        throw(TimeLimitException("BendersSeqInOut: Time limit reached during master solving"))
                    else
                        throw(UnexpectedModelStatusException("BendersSeqInOut: master $(termination_status(env.master.model))"))
                    end
                end
                
                # perturb point
                stabilizing_x = α * stabilizing_x + (1 - α) * state.values[:x]
                intermediate_x = λ * state.values[:x] + (1 - λ) * stabilizing_x

                # Execute oracle
                state.oracle_time = @elapsed begin
                    state.is_in_L, hyperplanes, state.f_x = generate_cuts(env.oracle, intermediate_x, state.values[:t]; time_limit = get_sec_remaining(log, param))

                    if kelley_mode 
                        if all(isfinite, state.f_x)
                            update_upper_bound_and_gap!(state, log, (f_x, x) -> env.master.c_t' * f_x + env.master.c_x' * x)
                        end
                    else
                        state.is_in_L = false
                    end

                    cuts = !state.is_in_L ? hyperplanes_to_expression(env.master.model, hyperplanes, env.master.x, env.master.t) : []
                end
            
                # Update state and record information
                record_iteration!(log, state)
            end

            param.verbose && print_iteration_info(state, log)

            # Check termination criteria
            is_terminated(state, log, param) && break

            # add generated cuts to master
            @constraint(env.master.model, 0 .>= cuts)
            
            # whether to switch kelley mode
            if !kelley_mode && log.n_iter != 0
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
        env.obj_value = log.iterations[end].LB
        
        return to_dataframe(log)
    catch e
        if e isa TimeLimitException
            @warn e.msg
            env.termination_status = TimeLimit()
            env.obj_value = isempty(log.iterations) ? Inf : log.iterations[end].LB
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
