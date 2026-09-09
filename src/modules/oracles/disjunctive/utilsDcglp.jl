
"""
    build_dcglp(
        master::AbstractMaster,
        normalization::AbstractNormalization,
        param::SplitOracleParam,
    )

Build the DCGLP used by [`SplitOracle`](@ref).

The method first constructs the common DCGLP formulation through
`build_dcglp_base`, then adds the constraint defined by `normalization`.

Returns the constructed JuMP model.
"""
function build_dcglp(
    master::AbstractMaster,
    normalization::AbstractNormalization,
    param::SplitOracleParam,
)
    dcglp, tau, sx, st = build_dcglp_base(master, param)

    add_normalization_constraint!(normalization, master, dcglp, tau, sx, st)

    return dcglp
end

"""
    build_dcglp_base(
        master::AbstractMaster,
        param::SplitOracleParam,
    )

Build the common DCGLP formulation shared by all normalization schemes.

The method creates the disjunctive variables for the two sides of the split, transfers supported linear constraints and bounds from the master problem, and introduces the auxiliary variables `tau`, `sx`, and `st` used by the normalization.

Normalization-specific constraints are not added by this method.

# Returns

A tuple `(dcglp, tau, sx, st)` containing the DCGLP model and the auxiliary
variables required to define its normalization.
"""
function build_dcglp_base(
    master::AbstractMaster,
    param::SplitOracleParam,
)
    dcglp = Model(param.dcglp_param.optimizer)
    @variable(dcglp, omega_0[1:2] >= 0)

    @variable(dcglp, omega_x[1:2, 1:master.dim_x])
    @variable(dcglp, omega_t[1:2, 1:master.dim_t])

    @constraint(dcglp, [i in 1:2], omega_t[i, :] .>= -1.0e6 .* omega_0[i])
    @constraint(dcglp, coneta[i in 1:2, j in 1:master.dim_x], 0 >= -omega_0[i] + omega_x[i, j])
    @constraint(dcglp, condelta[i in 1:2, j in 1:master.dim_x], 0 >= -omega_x[i, j])

    @constraint(dcglp, con0, omega_0[1] + omega_0[2] == 1)

    for i in 1:2
        transfer_scaled_linear_rows_and_bounds_with_types!(
            master.model,
            master.x,
            dcglp,
            omega_x[i, :],
            omega_0[i],
        )
    end

    @variable(dcglp, tau)
    @variable(dcglp, sx[1:master.dim_x])
    @variable(dcglp, st[1:master.dim_t])

    @objective(dcglp, Min, tau)

    @constraint(dcglp, conx, dcglp[:omega_x][1, :] + dcglp[:omega_x][2, :] - sx .== 0)
    @constraint(dcglp, cont[j = 1:master.dim_t], dcglp[:omega_t][1, j] + dcglp[:omega_t][2, j] - st[j] == 0)

    return dcglp, tau, sx, st
end

"""
    transfer_scaled_linear_rows_and_bounds_with_types!(master, x, dcglp, omega, omega0)

Transfer supported linear constraints and variable bounds from the master model to a scaled DCGLP representation.

Only constraints involving variables contained in `x` are transferred. Affine constraints and scalar variable bounds are scaled by `omega0` and expressed in terms of the corresponding variables in `omega`.

Unsupported constraint types are not transferred and generate a warning.
"""
function transfer_scaled_linear_rows_and_bounds_with_types!(
    master::Model,
    x::Vector{VariableRef},
    dcglp::Model,
    omega::Vector{VariableRef},
    omega0::VariableRef,
)
    pairs_present = JuMP.list_of_constraint_types(master)
    for (F, S) in pairs_present
        if F in [AffExpr; VariableRef]
            if S in [MOI.GreaterThan{Float64}; MOI.LessThan{Float64}; MOI.EqualTo{Float64}; MOI.Interval{Float64}]
                continue
            end
        end
        if S in [MOI.Integer; MOI.ZeroOne]
            continue
        end
        @warn "A master constraint of type ($F, $S) was not automatically incorporated into dcglp. If this constraint is linear, please add it manually."
    end

    idx_to_pos = Dict{Int,Int}()
    for (pos, v) in enumerate(x)
        vi = JuMP.index(v)
        idx_to_pos[vi.value] = pos
    end

    length(x) == length(omega) || error("x and omega must have the same length/structure.")

    backend = JuMP.backend(master)
    pair_types = [
        (MOI.VariableIndex, MOI.GreaterThan{Float64}),
        (MOI.VariableIndex, MOI.LessThan{Float64}),
        (MOI.VariableIndex, MOI.EqualTo{Float64}),
        (MOI.VariableIndex, MOI.Interval{Float64}),
        (MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}),
        (MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}),
        (MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}),
        (MOI.ScalarAffineFunction{Float64}, MOI.Interval{Float64}),
    ]

    for (F, S) in pair_types
        for ci in MOI.get(backend, MOI.ListOfConstraintIndices{F,S}())
            func = MOI.get(backend, MOI.ConstraintFunction(), ci)
            set = MOI.get(backend, MOI.ConstraintSet(), ci)

            terms = Tuple{Float64,Int}[]
            constant = 0.0

            if F == MOI.VariableIndex
                vpos = get(idx_to_pos, func.value, 0)
                vpos == 0 && continue
                push!(terms, (1.0, vpos))
            else
                constant = func.constant
                ok = true
                for term in func.terms
                    vpos = get(idx_to_pos, term.variable.value, 0)
                    if vpos == 0
                        ok = false
                        break
                    end
                    push!(terms, (term.coefficient, vpos))
                end
                ok || continue
            end

            expr = sum(a * omega[j] for (a, j) in terms)

            # MOI stores scalar affine constraints as `constant + expr in set`.
            # The lifted point (omega / omega0) must satisfy the same relation.
            if S == MOI.GreaterThan{Float64}
                @constraint(dcglp, expr + constant * omega0 >= set.lower * omega0)
            elseif S == MOI.LessThan{Float64}
                @constraint(dcglp, expr + constant * omega0 <= set.upper * omega0)
            elseif S == MOI.EqualTo{Float64}
                @constraint(dcglp, expr + constant * omega0 == set.value * omega0)
            elseif S == MOI.Interval{Float64}
                @constraint(dcglp, expr + constant * omega0 <= set.upper * omega0)
                @constraint(dcglp, expr + constant * omega0 >= set.lower * omega0)
            end
        end
    end
end

"""
    solve_dcglp!(
        oracle::SplitOracle,
        x_value::Vector{Float64},
        t_value::Vector{Float64},
        zero_indices::Vector{Int},
        one_indices::Vector{Int};
        time_limit::Float64,
    )

Execute the DCGLP cutting-plane separation loop for a candidate master solution.

At each iteration, the method solves the current DCGLP relaxation, evaluates the typical oracles at the two disjunctive points, adds generated Benders cuts to the DCGLP, and updates the DCGLP bounds and termination state.

If typical-oracle cut generation is interrupted by a time limit or an unexpected model status, the current DCGLP solution is retained and no incomplete Benders cuts are added to the DCGLP.

After the loop terminates, a disjunctive cut is constructed if the current DCGLP lower bound is sufficiently positive and a valid dual solution is available. If no such cut can be constructed but both disjunctive points belong to their respective typical-oracle feasible regions, the candidate is reported as belonging to the split oracle's feasible region. Otherwise, separation falls back to the first typical oracle.

An unexpected model status from the DCGLP itself may also trigger fallback to the typical oracle, controlled by `oracle.param.fallback_to_typical_cuts`.

# Returns

A tuple `(is_in_L, hyperplanes, sub_obj_vals)` as specified by [`generate_cuts`](@ref).

# Error Handling

- A time limit or unexpected model status encountered during typical-oracle cut generation terminates the DCGLP loop while preserving the current DCGLP solution.
- An unexpected model status returned by the DCGLP master triggers fallback to the first typical oracle when `fallback_to_typical_cuts` is enabled.
- Other exceptions are propagated to the caller.
"""
function solve_dcglp!(
    oracle::SplitOracle,
    x_value::Vector{Float64},
    t_value::Vector{Float64},
    zero_indices::Vector{Int},
    one_indices::Vector{Int};
    time_limit::Float64,
)
    log = DcglpLog()
    normalization = oracle.normalization

    dcglp = oracle.dcglp
    hyperplanes = Hyperplane[]

    try
        while true
            get_sec_remaining(log.start_time, time_limit) <= 0.0 && break

            state = DcglpState()
            benders_cuts = Dict(1 => AffExpr[], 2 => AffExpr[])
            
            state.total_time = @elapsed begin
                # Solve dcglp relaxation
                state.master_time = @elapsed begin
                    set_time_limit_sec(dcglp, get_sec_remaining(log.start_time, time_limit))
                    optimize!(dcglp)
                    if is_solved_and_feasible(dcglp; allow_local = false, dual = true)
                        read_dcglp_solution!(oracle, state)
                    elseif termination_status(dcglp) == TIME_LIMIT
                        break
                    else
                        throw(UnexpectedModelStatusException("SplitOracle: DCGLP master terminated with $(termination_status(dcglp))."))
                    end
                end 
                    
                # Execute oracle
                try
                    collect_dcglp_benders_cuts!(oracle, state, benders_cuts, hyperplanes, x_value, t_value, log, time_limit)
                catch err
                    if err isa TimeLimitException || err isa UnexpectedModelStatusException
                        @warn "SplitOracle: typical-oracle cut generation was interrupted " *
                              "($(err.msg)); using the current DCGLP solution."
                        break
                    end
                    rethrow()
                end
                update_dcglp_upper_bound_and_gap!(normalization, state, log, t_value)
                record_iteration!(log, state)
            end

            oracle.param.dcglp_param.verbose && print_iteration_info(state, log)
            check_lb_improvement!(state, log; zero_tol = oracle.param.zero_tol)
            is_terminated(state, log, oracle.param.dcglp_param) && break

            cuts_to_add = [benders_cuts[1]; benders_cuts[2]]
            !isempty(cuts_to_add) && add_constraints(dcglp, :con_benders, cuts_to_add)
        end

        if isempty(log.iterations)
            @warn "SplitOracle: DCGLP cutting-plane loop terminated without any iterations."
            return generate_cuts(oracle.typical_oracles[1], x_value, t_value; time_limit = get_sec_remaining(log.start_time, time_limit))
        end

        # If the final DCGLP lower bound is sufficiently positive, construct a disjunctive cut. Otherwise, fallback to a typical oracle.
        current_lb = log.iterations[end].LB
        if current_lb >= oracle.param.zero_tol && has_duals(dcglp)
            cut = build_dcglp_disjunctive_cut(
                normalization,
                dcglp,
                oracle.param,
                zero_indices,
                one_indices,
            )
            oracle.param.dcglp_param.verbose && print_disjunctive_cut(oracle, cut, x_value, t_value; zero_tol = oracle.param.zero_tol)
            store_dcglp_disjunctive_cut!(oracle, cut, hyperplanes)
            return false, hyperplanes, fill(Inf, length(t_value))
        end

        if all(log.iterations[end].is_in_L) # optimal termination with both points in the oracle feasible region
            return true, [Hyperplane(length(x_value), length(t_value))], deepcopy(t_value)
        end

        # fallback to typical oracle since no meaningful disjunctive cut can be constructed from the DCGLP solution
        return generate_cuts(oracle.typical_oracles[1], x_value, t_value; time_limit = get_sec_remaining(log.start_time, time_limit))

    catch e
        if e isa UnexpectedModelStatusException
            @warn "$SplitOracle: DCGLP cutting-plane loop terminated with unexpected dcglp model status"
            if oracle.param.fallback_to_typical_cuts
                @warn "$SplitOracle: fallback to typical oracle due to DCGLP error"
                return generate_cuts(oracle.typical_oracles[1], x_value, t_value; time_limit = get_sec_remaining(log.start_time, time_limit))
            end
        else
            rethrow()
        end
    end
end

function read_dcglp_solution!(oracle::SplitOracle, state::DcglpState)
    dcglp = oracle.dcglp
    for i in 1:2
        state.values[:ω_x][i] = value.(dcglp[:omega_x][i, :])
        state.values[:ω_t][i] = value.(dcglp[:omega_t][i, :])
        state.values[:ω_0][i] = value(dcglp[:omega_0][i])
    end
    tau_value = value(dcglp[:tau])
    state.values[:tau] = tau_value
    state.values[:sx] = value.(dcglp[:sx])
    state.LB = tau_value
end

"""
    collect_dcglp_benders_cuts!(
        oracle,
        state,
        benders_cuts,
        hyperplanes,
        x_value,
        t_value,
        log,
        time_limit,
    )

Evaluate the typical oracles at the two DCGLP disjunctive points and collect the resulting Benders cuts.

Cuts violated by the DCGLP points are converted to the scaled DCGLP representation and added to `benders_cuts`. Depending on `oracle.param.add_benders_cuts_to_master`, selected byproduct Benders cuts may
also be appended to `hyperplanes` for addition to the master problem.
"""
function collect_dcglp_benders_cuts!(
    oracle::SplitOracle,
    state::DcglpState,
    benders_cuts::Dict{Int, Vector{AffExpr}},
    hyperplanes::Vector{Hyperplane},
    x_value::Vector{Float64},
    t_value::Vector{Float64},
    log::DcglpLog,
    time_limit::Float64,
)
    dcglp = oracle.dcglp
    for i in 1:2
        state.oracle_times[i] = @elapsed begin
            if state.values[:ω_0][i] >= oracle.param.zero_tol
                t_prime = state.values[:ω_t][i] ./ state.values[:ω_0][i]
                state.is_in_L[i], hyperplanes_i, state.f_x[i] = generate_cuts(
                    oracle.typical_oracles[i],
                    clamp.(state.values[:ω_x][i] ./ state.values[:ω_0][i], 0.0, 1.0),
                    t_prime;
                    tol_normalize = state.values[:ω_0][i],
                    time_limit = get_sec_remaining(log.start_time, time_limit),
                )

                if !state.is_in_L[i]
                    for k in 1:2
                        append!(
                            benders_cuts[i],
                            hyperplanes_to_expression(
                                dcglp,
                                hyperplanes_i,
                                dcglp[:omega_x][k, :],
                                dcglp[:omega_t][k, :],
                                dcglp[:omega_0][k],
                            ),
                        )
                    end
                    append_selected_benders_cuts_to_master!(
                        oracle, hyperplanes, hyperplanes_i, x_value, t_value,
                    )
                end
            else
                state.is_in_L[i] = true
            end
        end
    end
end

function append_selected_benders_cuts_to_master!(
    oracle::SplitOracle,
    hyperplanes::Vector{Hyperplane},
    candidate_hyperplanes::Vector{Hyperplane},
    x_value::Vector{Float64},
    t_value::Vector{Float64},
)
    oracle.param.add_benders_cuts_to_master == 0 && return nothing

    add_violated = oracle.param.add_benders_cuts_to_master == 2
    append!(
        hyperplanes,
        select_top_fraction(
            candidate_hyperplanes,
            h -> evaluate_violation(h, x_value, t_value),
            oracle.param.fraction_of_benders_cuts_to_master;
            add_only_violated_cuts = add_violated,
        ),
    )
end
