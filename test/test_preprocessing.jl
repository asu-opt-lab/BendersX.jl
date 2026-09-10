using Test
using JuMP

struct PreprocessingTestOracle <: BendersX.AbstractOracle end
struct PreprocessingTestTypicalOracle <: BendersX.AbstractTypicalOracle end
struct PreprocessingTestDisjunctiveOracle <: BendersX.AbstractDisjunctiveOracle end

struct PreprocessingTestMaster <: BendersX.AbstractMaster
    model::Model
end

struct PreprocessingProbeException <: Exception end
struct ThrowingPreprocessing <: BendersX.AbstractPreprocessing end

BendersX.preprocess!(::BendersX.AbstractMaster, ::ThrowingPreprocessing; time_limit::Float64 = 100.0) =
    throw(PreprocessingProbeException())

struct PreprocessingRecordingSeq <: BendersX.AbstractBendersSeq
    master::BendersX.AbstractMaster
    oracle::BendersX.AbstractOracle
    param::BendersX.AbstractBendersSeqParam
end

PreprocessingRecordingSeq(
    master::BendersX.AbstractMaster,
    oracle::BendersX.AbstractOracle;
    param::BendersX.AbstractBendersSeqParam,
) = PreprocessingRecordingSeq(master, oracle, param)

const preprocessing_seq_calls = Any[]
const throw_during_disjunctive_preprocessing = Ref(false)

function BendersX.solve!(env::PreprocessingRecordingSeq)
    variable = only(all_variables(env.master.model))
    push!(preprocessing_seq_calls, (
        oracle = env.oracle,
        param = env.param,
        time_limit = env.param.time_limit,
        is_binary = is_binary(variable),
    ))

    if throw_during_disjunctive_preprocessing[] &&
       env.oracle isa PreprocessingTestDisjunctiveOracle
        throw(PreprocessingProbeException())
    end

    return nothing
end

function preprocessing_test_binary_master()
    model = Model()
    variable = @variable(model, binary = true)
    return PreprocessingTestMaster(model), variable
end

@testset "Sequential preprocessing" begin
    @testset "LPRelaxationPreprocessing keyword constructor" begin
        oracle = PreprocessingTestOracle()
        param = BendersSeqParam(verbose = false)

        keyword = LPRelaxationPreprocessing(
            oracle;
            seq_env_type = BendersSeq,
            param = param,
        )

        @test keyword.oracle === oracle
        @test keyword.seq_env_type === BendersSeq
        @test keyword.param === param
        @test_throws MethodError LPRelaxationPreprocessing(oracle, BendersSeq, param)
    end

    @testset "DisjunctiveLPRelaxationPreprocessing constructor" begin
        typical_oracle = PreprocessingTestTypicalOracle()
        disjunctive_oracle = PreprocessingTestDisjunctiveOracle()
        param = BendersSeqParam(verbose = false)

        preprocessing = DisjunctiveLPRelaxationPreprocessing(
            typical_oracle,
            disjunctive_oracle;
            seq_env_type = PreprocessingRecordingSeq,
            param = param,
        )

        @test preprocessing.typical_oracle === typical_oracle
        @test preprocessing.disjunctive_oracle === disjunctive_oracle
        @test preprocessing.seq_env_type === PreprocessingRecordingSeq
        @test preprocessing.param === param
    end

    @testset "DisjunctiveLPRelaxationPreprocessing two-phase execution" begin
        empty!(preprocessing_seq_calls)
        throw_during_disjunctive_preprocessing[] = false
        master, variable = preprocessing_test_binary_master()
        typical_oracle = PreprocessingTestTypicalOracle()
        disjunctive_oracle = PreprocessingTestDisjunctiveOracle()
        param = BendersSeqParam(time_limit = 10.0, verbose = false)
        preprocessing = DisjunctiveLPRelaxationPreprocessing(
            typical_oracle,
            disjunctive_oracle;
            seq_env_type = PreprocessingRecordingSeq,
            param = param,
        )

        elapsed = BendersX.preprocess!(master, preprocessing)

        @test elapsed >= 0.0
        @test length(preprocessing_seq_calls) == 2
        @test preprocessing_seq_calls[1].oracle === typical_oracle
        @test preprocessing_seq_calls[2].oracle === disjunctive_oracle
        @test all(!call.is_binary for call in preprocessing_seq_calls)
        @test preprocessing_seq_calls[1].param === preprocessing_seq_calls[2].param
        @test preprocessing_seq_calls[1].param !== param
        @test preprocessing_seq_calls[1].time_limit == 10.0
        @test 0.0 <= preprocessing_seq_calls[2].time_limit <= 10.0
        @test param.time_limit == 10.0
        @test is_binary(variable)
    end

    @testset "DisjunctiveLPRelaxationPreprocessing restores integrality on error" begin
        empty!(preprocessing_seq_calls)
        throw_during_disjunctive_preprocessing[] = true
        master, variable = preprocessing_test_binary_master()
        preprocessing = DisjunctiveLPRelaxationPreprocessing(
            PreprocessingTestTypicalOracle(),
            PreprocessingTestDisjunctiveOracle();
            seq_env_type = PreprocessingRecordingSeq,
            param = BendersSeqParam(verbose = false),
        )

        @test_throws PreprocessingProbeException BendersX.preprocess!(master, preprocessing)
        @test length(preprocessing_seq_calls) == 2
        @test is_binary(variable)
        throw_during_disjunctive_preprocessing[] = false
    end

    @testset "BendersSeqInOut preprocessing" begin
        preprocessing = ThrowingPreprocessing()
        env = BendersSeqInOut(
            PreprocessingTestMaster(Model()),
            PreprocessingTestOracle();
            param = BendersSeqInOutParam(
                stabilizing_x = Float64[],
                verbose = false,
            ),
            preprocessing = preprocessing,
        )

        @test env.preprocessing === preprocessing
        @test_throws PreprocessingProbeException solve!(env)
    end
end
