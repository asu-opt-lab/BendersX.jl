"""
    LPRelaxationPreprocessing <: AbstractPreprocessing

LP-relaxation preprocessing using a sequential Benders environment.

This strategy temporarily relaxes the integrality constraints of the master problem and runs the configured sequential Benders environment to generate initial cuts. The original integrality constraints are restored after preprocessing.

# Fields

- `oracle::AbstractOracle`: Oracle used to generate cuts during preprocessing.
- `seq_env_type::Type{<:AbstractBendersSeq}`: Sequential Benders environment type used for preprocessing.
- `param::AbstractBendersSeqParam`: Parameters passed to the sequential preprocessing environment.

# Constructor

    LPRelaxationPreprocessing(
        oracle::AbstractOracle;
        seq_env_type::Type{<:AbstractBendersSeq} = BendersSeq,
        param::AbstractBendersSeqParam = BendersSeqParam(),
    )

Construct an LP-relaxation preprocessing strategy using `oracle`.

By default, [`BendersSeq`](@ref) is used as the sequential preprocessing environment with default [`BendersSeqParam`](@ref) parameters.
"""
mutable struct LPRelaxationPreprocessing <: AbstractPreprocessing
    oracle::AbstractOracle
    seq_env_type::Type{<:AbstractBendersSeq}
    param::AbstractBendersSeqParam

    function LPRelaxationPreprocessing(oracle::AbstractOracle; seq_env_type::Type{<:AbstractBendersSeq} = BendersSeq, param::AbstractBendersSeqParam = BendersSeqParam())
        new(oracle, seq_env_type, param)
    end
end

"""
    preprocess!(
        master::AbstractMaster,
        preprocessing::LPRelaxationPreprocessing,
    )

Apply LP-relaxation preprocessing to `master`.

The method temporarily relaxes the integrality constraints of the master problem and executes the sequential Benders environment specified by `preprocessing`. Cuts generated during this solve remain in the master model, while the original integrality constraints are restored before returning, including when preprocessing terminates with an error.

A private copy of the preprocessing parameters is used because the preprocessing solve adjusts the nested time limit without modifying the configured parameter object.

# Arguments

- `master::AbstractMaster`: Master problem to preprocess.
- `preprocessing::LPRelaxationPreprocessing`: LP-relaxation preprocessing environment.

# Returns

The elapsed preprocessing time in seconds.
"""
function preprocess!(master::AbstractMaster, preprocessing::LPRelaxationPreprocessing; time_limit::Float64 = 100.0)
   
    tic = time()

    seq_param = deepcopy(preprocessing.param)
    seq_param.time_limit = max(0.0, min(time_limit, seq_param.time_limit))
    
    # Relax integrality, ensure undo() always runs even on error
    undo = relax_integrality(master.model)
    
    # measure time and ensure undo() is called even if solve! errors
    try
        env_preprocessing = preprocessing.seq_env_type(master, preprocessing.oracle; param = seq_param)
        solve!(env_preprocessing)
    finally
        # always restore integrality (even on exceptions)
        undo()
    end
    
    return time() - tic
end
