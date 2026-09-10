using Test
using BendersX

@testset "Public API Contract" begin
    exported = [
        :Master,
        :BendersSeq, :BendersSeqInOut, :BendersBnB,
        :ClassicalOracle, :UnifiedOracle, :ParetoOracle, :SeparableOracle,
        :SplitOracle,
        :solve!,
        :BasicOracleParam, :ClassicalOracleParam, :UnifiedOracleParam, :ParetoOracleParam,
        :SeparableOracleParam, :DcglpParam,
        :SplitOracleParam, :LpDistanceNormalization, :ReversePolarNormalization,
        :BendersSeqParam, :BendersSeqInOutParam, :BendersBnBParam,
        :LazyCallback, :UserCallback, :NoUserCallback, :UserCallbackParam,
        :LPRelaxationPreprocessing, :NoPreprocessing, :DisjunctiveLPRelaxationPreprocessing,
        :TerminationStatus, :NotSolved, :TimeLimit, :Optimal, :InfeasibleOrNumericalIssue,
        :GBCBoundType, :UpperBound, :LowerBound, :Fixed,
        :RandomFractional, :MostFractional, :LargestFractional,
        :NoDisjunctiveCuts, :AllDisjunctiveCuts, :DisjunctiveCutsSmallerIndices,
        :update_master_model!, :update_sub_model!,
        :CFLPData, :UFLPData, :SCFLPData, :SNIPData,
        :CFLKnapsackOracle, :CFLKnapsackOracleParam, :UFLKnapsackOracle, :UFLKnapsackOracleParam,
        :read_GK_data, :read_cfl_file, :read_cflp_benchmark_data,
        :read_uflp_benchmark_data, :read_Simple_data,
        :read_stochastic_capacited_facility_location_problem, :read_snip_data,
    ]

    public_only = [
        :AbstractBendersEnv, :AbstractBendersSeq, :AbstractBendersBnB,
        :AbstractMaster, :AbstractOracle, :AbstractOracleParam, :AbstractTypicalOracle,
        :AbstractDisjunctiveOracle,
        :AbstractPreprocessing,
        :AbstractNormalization,
        :AbstractLazyCallback, :AbstractUserCallback,
        :AbstractLoopState, :AbstractLoopLog, :AbstractLoopParam,
        :AbstractBendersSeqState, :AbstractBendersSeqLog, :AbstractBendersSeqParam,
        :AbstractBendersBnBState, :AbstractBendersBnBLog, :AbstractBendersBnBParam,
        :SplitIndexSelectionRule, :DisjunctiveCutsAppendRule,
        :generate_cuts, :add_normalization_constraint!, :update_dcglp_upper_bound_and_gap!,
        :disjunctive_cut_normalization_value, :Hyperplane, :aggregate, :evaluate_violation,
        :select_top_fraction, :hyperplanes_to_expression, :add_constraints,
        :copy_variables!, :var_from_tuple, :transfer_scaled_linear_rows_and_bounds_with_types!,
        :infeasibility_report,
        :TimeLimitException, :UnexpectedModelStatusException, :UnimplementedInterfaceException,
        :AlgorithmException, :UnsupportedModelException,
    ]

    private_symbols = [
        :lazy_callback, :user_callback, :preprocess!,
        :get_sec_remaining, :record_iteration!, :update_upper_bound_and_gap!,
        :is_terminated, :check_lb_improvement!, :print_iteration_info, :to_dataframe,
        :calculate_KP_value,
        :BendersSeqState, :BendersSeqLog, :BendersBnBState, :BendersBnBLog,
        Symbol("Abstract" * "Dcglp" * "Oracle"),
        Symbol("Dcglp" * "Oracle"),
        Symbol("Dcglp" * "OracleParam"),
    ]

    visible_names = Set(names(BendersX))

    removed_data_supertype = Symbol("Abstract", "Data")
    @test !isdefined(BendersX, removed_data_supertype)
    @test !(removed_data_supertype in visible_names)
    @test !isdefined(BendersX, :set_parameter!)

    for sym in exported
        @test Base.isexported(BendersX, sym)
        @test Base.ispublic(BendersX, sym)
        @test sym in visible_names
    end

    for sym in public_only
        @test !Base.isexported(BendersX, sym)
        @test Base.ispublic(BendersX, sym)
        @test sym in visible_names
    end

    for sym in private_symbols
        @test !Base.ispublic(BendersX, sym)
        @test !(sym in visible_names)
    end
end
