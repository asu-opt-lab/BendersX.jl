# ----------------------------------------------------------------------------
# Callback infrastructure
# ----------------------------------------------------------------------------

include("callback/callback.jl") # must be included first
 
"""
    BendersBnB <: AbstractBendersBnB

Branch-and-bound Benders decomposition environment.

`BendersBnB` integrates Benders cut generation into the MIP solver's branch-and-bound process through lazy-constraint and optional user-cut callbacks. Integral candidate solutions are processed by a lazy-constraint callback, while an optional user-cut callback can separate fractional candidates. A preprocessing procedure can also be applied to the
master before branch-and-bound begins.

# Fields
- `master::AbstractMaster`: Master problem.
- `param::BendersBnBParam`: Parameters controlling algorithm, such as the time limit, relative gap tolerance, and verbosity.
- `preprocessing::AbstractPreprocessing`: Preprocessing procedure applied before branch-and-bound.
- `lazy_callback::AbstractLazyCallback`: Configuration for lazy-constraint callback
- `user_callback::AbstractUserCallback`: Configuration for user-cut callback
- `obj_value::Float64`: Objective value of the best solution found
- `termination_status::TerminationStatus`: Termination status of the Benders algorithm. See [`TerminationStatus`](@ref) for available statuses.

# Constructors

    BendersBnB(
        master::AbstractMaster;
        lazy_callback::AbstractLazyCallback,
        preprocessing::AbstractPreprocessing = NoPreprocessing(),
        user_callback::AbstractUserCallback = NoUserCallback(),
        param::BendersBnBParam = BendersBnBParam(),
    )

Construct a `BendersBnB` environment from explicitly configured callback and preprocessing procedures. `lazy_callback` is required because it determines how integral candidates encountered during branch-and-bound are separated. Preprocessing and user-cut separation are optional.

    BendersBnB(
        master::AbstractMaster,
        oracle::AbstractOracle;
        preprocessing::AbstractPreprocessing = NoPreprocessing(),
        param::BendersBnBParam = BendersBnBParam(),
    )

Convenience constructor for the common configuration. It constructs `LazyCallback(oracle)` for integral-candidate separation and uses `NoUserCallback()` for fractional candidates. The supplied `oracle` is used to construct the lazy callback; it is not stored separately by `BendersBnB`.

# Examples

A basic configuration uses an oracle for lazy separation, with no preprocessing or user callback:

```julia
master = Master(data; model = update_master_model!)
oracle = ClassicalOracle(data, master; model = update_sub_model!)
env = BendersBnB(master, oracle)
result = solve!(env)
```

For explicit control of callback behavior, construct the callbacks separately:
```julia
lazy_callback = LazyCallback(classical_oracle)

user_callback = UserCallback(
    disjunctive_oracle;
    param = UserCallbackParam(frequency = 100),
)

env = BendersBnB(
    master;
    lazy_callback = lazy_callback,
    preprocessing = preprocessing,
    user_callback = user_callback,
    param = bnb_param,
)

result = solve!(env)
```

The lazy and user callbacks may use different oracles.

See also: [`BendersSeq`](@ref), [`AbstractPreprocessing`](@ref), [`AbstractLazyCallback`](@ref), [`AbstractUserCallback`](@ref), [`LazyCallback`](@ref), [`UserCallback`](@ref)
"""
mutable struct BendersBnB <: AbstractBendersBnB
    master::AbstractMaster 

    param::BendersBnBParam 

    preprocessing::AbstractPreprocessing
    lazy_callback::AbstractLazyCallback
    user_callback::AbstractUserCallback

    obj_value::Float64 
    termination_status::TerminationStatus 

    function BendersBnB(
        master::AbstractMaster;
        lazy_callback::AbstractLazyCallback,
        preprocessing::AbstractPreprocessing = NoPreprocessing(),
        user_callback::AbstractUserCallback = NoUserCallback(),
        param::BendersBnBParam = BendersBnBParam(),
    )
        new(master, param, preprocessing, lazy_callback, user_callback, Inf, NotSolved())
    end
end

function BendersBnB(
    master::AbstractMaster,
    oracle::AbstractOracle;
    preprocessing::AbstractPreprocessing = NoPreprocessing(),
    param::BendersBnBParam = BendersBnBParam(),
)
    BendersBnB(
        master;
        lazy_callback = LazyCallback(oracle),
        preprocessing = preprocessing,
        user_callback = NoUserCallback(),
        param = param
    )
end

"""
    solve!(env::BendersBnB) -> DataFrame

Execute the branch-and-bound Benders decomposition algorithm.

The method first applies the configured preprocessing, then installs the configured lazy-constraint and user-cut callbacks, sets the solver parameters, and solves the master problem through the MIP solver's branch-and-bound procedure.

The returned `DataFrame` contains summary statistics for the solve.

# Arguments
- `env::BendersBnB`: The configured Benders Branch-and-Bound environment

# Returns
A one-row `DataFrame` containing the solve statistics, including the objective value, objective bound, elapsed time, relative gap, and cut-generation counts. The objective value is Inf when no feasible solution is available.

# Termination
The environment's `obj_value` and `termination_status` fields are updated before returning. A time limit or unexpected solver status is handled and reflected in the returned environment and solve statistics.
"""
function solve!(env::BendersBnB) 
    log = BendersBnBLog()
    param = env.param
    try 
        # Apply preprocessing
        log.preprocessing_time = preprocess!(env.master, env.preprocessing; time_limit = get_sec_remaining(log, param))

        # Set up lazy callback
        function lazy_callback_wrapper(cb_data)
            lazy_callback(cb_data, env.master, log, env.param, env.lazy_callback)
        end
        set_attribute(env.master.model, MOI.LazyConstraintCallback(), lazy_callback_wrapper)
        
        # Set up user callback if specified
        if !isa(env.user_callback, NoUserCallback)
            function user_callback_wrapper(cb_data)
                user_callback(cb_data, env.master, log, env.param, env.user_callback)
            end
            set_attribute(env.master.model, MOI.UserCutCallback(), user_callback_wrapper)
        end
        
        get_sec_remaining(log, param) <= 0.0 && throw(TimeLimitException("BendersBnB: Time limit reached before initiating branch-and-bound procedure."))

        # Configure solver parameters
        set_time_limit_sec(env.master.model, get_sec_remaining(log, param))
        set_optimizer_attribute(env.master.model, MOI.Silent(), !param.verbose)
        set_optimizer_attribute(env.master.model, MOI.RelativeGapTolerance(), param.gap_tolerance)
        
        # Solve the master problem
        JuMP.optimize!(env.master.model)
        
        log.total_time = time() - log.start_time

        # Process termination status
        status = termination_status(env.master.model)
        if status == MOI.OPTIMAL
            env.termination_status = Optimal()
            env.obj_value = JuMP.objective_value(env.master.model)
        elseif status == MOI.TIME_LIMIT
            env.termination_status = TimeLimit()
            env.obj_value = has_values(env.master.model) ? JuMP.objective_value(env.master.model) : Inf
        else
            throw(UnexpectedModelStatusException("BendersBnB: master $(status)"))
        end
       
        df = to_dataframe(env, log)

        if param.verbose 
            @info df
        end
        
        return df
    catch e
        if e isa TimeLimitException
            @warn e.msg
            env.termination_status = TimeLimit()
            env.obj_value = has_values(env.master.model) ? JuMP.objective_value(env.master.model) : Inf
        elseif e isa UnexpectedModelStatusException
            @warn e.msg
            env.termination_status = InfeasibleOrNumericalIssue()
        else
            rethrow()  
        end
        if env.param.verbose
            println("BendersBnB: Terminated with $(env.termination_status)")
        end
        return to_dataframe(env, log)
    end
end
