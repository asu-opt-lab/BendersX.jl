function add_disjunctive_cuts!(oracle::AbstractDisjunctiveOracle, rule::DisjunctiveCutsAppendRule)
    throw(UnimplementedInterfaceException("update add_disjunctive_cuts! for $(typeof(rule))"))
end

function add_disjunctive_cuts!(oracle::AbstractDisjunctiveOracle, ::NoDisjunctiveCuts)
end

function add_disjunctive_cuts!(oracle::AbstractDisjunctiveOracle, ::AllDisjunctiveCuts)
    install_disjunctive_cuts!(oracle, oracle.disjunctive_cuts)
end

function add_disjunctive_cuts!(oracle::AbstractDisjunctiveOracle, ::DisjunctiveCutsSmallerIndices)
    oracle.param.split_index_selection_rule isa SimpleSplit ||
        throw(AlgorithmException("add_disjunctive_cuts!: DisjunctiveCutsSmallerIndices requires a simple split rule."))

    index = get_split_index(oracle)
    cuts = if index > 1
        reduce(vcat, oracle.disjunctive_cuts_by_index[1:index-1]; init = Hyperplane[])
    else
        Hyperplane[]
    end
    install_disjunctive_cuts!(oracle, cuts)
end

function install_disjunctive_cuts!(oracle::AbstractDisjunctiveOracle, cuts::Vector{Hyperplane})
    dcglp = oracle.dcglp
    isempty(cuts) && return nothing

    exprs = AffExpr[]
    for k in 1:2
        append!(
            exprs,
            hyperplanes_to_expression(
                dcglp,
                cuts,
                dcglp[:omega_x][k, :],
                dcglp[:omega_t][k, :],
                dcglp[:omega_0][k],
            ),
        )
    end
    add_constraints(dcglp, :con_disjunctive, exprs)
end

function update_dynamic_dcglp_constraints!(oracle::AbstractDisjunctiveOracle)
    if !oracle.param.reuse_dcglp
        delete_registered_constraints!(oracle.dcglp, :con_benders)
    end
    delete_registered_constraints!(oracle.dcglp, :con_disjunctive)
    add_disjunctive_cuts!(oracle, oracle.param.disjunctive_cut_append_rule)
end

function store_dcglp_disjunctive_cut!(
    oracle::AbstractDisjunctiveOracle,
    cut::Hyperplane,
    master_hyperplanes::Vector{Hyperplane},
)
    push!(oracle.disjunctive_cuts, cut)
    
    if oracle.param.split_index_selection_rule isa SimpleSplit
        index = get_split_index(oracle)
        push!(oracle.disjunctive_cuts_by_index[index], cut)
    end

    push!(master_hyperplanes, cut)
end

"""
    apply_lift_or_strengthen(dcglp, gamma_x, zero_indices, one_indices; lift, strengthen, zero_tol, gamma_0=0.0)

Apply lifting (preferred if `lift` and any zero/one index exists) or pure
strengthening to the cut coefficients. Returns the (possibly modified)
`(gamma_x, gamma_0)` pair.
"""
function apply_lift_or_strengthen(
    dcglp::Model,
    gamma_x::Vector{Float64},
    zero_indices::Vector{Int},
    one_indices::Vector{Int};
    lift::Bool,
    strengthen::Bool,
    zero_tol::Float64,
    gamma_0::Float64 = 0.0,
)
    if lift && (!isempty(zero_indices) || !isempty(one_indices))
        return lift_x_coefficients(
            dcglp, gamma_x, gamma_0, zero_indices, one_indices;
            strengthen = strengthen, zero_tol = zero_tol,
        )
    elseif strengthen
        sigma, delta = read_strengthening_duals(dcglp)
        return -strengthen_coefficients(-gamma_x, sigma, delta; zero_tol = zero_tol), gamma_0
    end
    return gamma_x, gamma_0
end

read_strengthening_duals(dcglp::Model) = (
    Dict(1 => dual(dcglp[:con_split_kappa]), 2 => dual(dcglp[:con_split_nu])),
    Dict(1 => dual.(dcglp[:condelta][1, :]), 2 => dual.(dcglp[:condelta][2, :])),
)

"""
    lift_x_coefficients(dcglp, gamma_x, gamma_0, zero_indices, one_indices; strengthen, zero_tol)

Apply lifting (and optional strengthening) to `gamma_x`. Returns the lifted
`(gamma_x, gamma_0)` pair in the common cut orientation.
"""
function lift_x_coefficients(
    dcglp::Model,
    gamma_x::Vector{Float64},
    gamma_0::Float64,
    zero_indices::Vector{Int},
    one_indices::Vector{Int};
    strengthen::Bool,
    zero_tol::Float64,
)
    zeta_k = !isempty(zero_indices) ? dual.(dcglp[:con_zeta][1, :]) : Float64[]
    zeta_v = !isempty(zero_indices) ? dual.(dcglp[:con_zeta][2, :]) : Float64[]
    xi_k = !isempty(one_indices) ? dual.(dcglp[:con_xi][1, :]) : Float64[]
    xi_v = !isempty(one_indices) ? dual.(dcglp[:con_xi][2, :]) : Float64[]

    lifted_gamma_0 = gamma_0 - sum(max.(xi_k, xi_v))
    lifted_gamma_x = -gamma_x
    lifted_gamma_x[zero_indices] .= -gamma_x[zero_indices] .+ max.(zeta_k, zeta_v)
    lifted_gamma_x[one_indices] .= -gamma_x[one_indices] .- max.(xi_k, xi_v)

    if strengthen
        sigma, base_delta = read_strengthening_duals(dcglp)
        delta_1 = copy(base_delta[1])
        delta_2 = copy(base_delta[2])
        delta_1[zero_indices] .+= -zeta_k .+ max.(zeta_k, zeta_v)
        delta_2[zero_indices] .+= -zeta_v .+ max.(zeta_k, zeta_v)
        lifted_gamma_x = strengthen_coefficients(lifted_gamma_x, sigma, Dict(1 => delta_1, 2 => delta_2); zero_tol = zero_tol)
    end

    return -lifted_gamma_x, lifted_gamma_0
end

function strengthen_coefficients(gamma_x, sigma, delta; zero_tol = 1.0e-9)
    a_1 = gamma_x .- delta[1]
    a_2 = gamma_x .- delta[2]
    sigma_sum = sigma[1] + sigma[2]
    sigma_sum < zero_tol && return copy(gamma_x)
    m = (a_1 .- a_2) ./ sigma_sum
    return min.(a_1 .- sigma[1] .* floor.(m), a_2 .+ sigma[2] .* ceil.(m))
end

function build_dcglp_disjunctive_cut(
    normalization::AbstractNormalization,
    dcglp::Model,
    common::SplitOracleParam,
    zero_indices::Vector{Int},
    one_indices::Vector{Int},
)
    gamma_x = dual.(dcglp[:conx])
    gamma_t = dual.(dcglp[:cont])
    gamma_0 = dual(dcglp[:con0])

    gamma_x, gamma_0 = apply_lift_or_strengthen(
        dcglp, gamma_x, zero_indices, one_indices;
        lift = common.lift, strengthen = common.strengthened,
        zero_tol = common.zero_tol, gamma_0 = gamma_0,
    )

    norm_value = disjunctive_cut_normalization_value(
        normalization,
        dcglp,
        gamma_x,
        gamma_t,
    )
    return Hyperplane(gamma_x ./ norm_value, gamma_t ./ norm_value, gamma_0 / norm_value)
end
