
"""
    DisjunctiveLPRelaxationPreprocessing <: AbstractPreprocessing

Two-phase preprocessing that first runs a typical
oracle-based separation and then a disjunctive oracle separation to
produce stronger initial cuts.
# Fields
- `typical_oracle::AbstractTypicalOracle`: Oracle for the first phase (classical preprocessing)
- `disjunctive_oracle::AbstractDisjunctiveOracle`: Oracle for the second phase
- `seq_env_type::Type{<:AbstractBendersSeq}`: Type of sequential Benders algorithm to use
- `param::AbstractBendersSeqParam`: Parameters for the sequential algorithm

# Constructor
```julia
DisjunctiveLPRelaxationPreprocessing(
    typical_oracle::AbstractTypicalOracle,
    disjunctive_oracle::AbstractDisjunctiveOracle;
    seq_env_type::Type{<:AbstractBendersSeq} = BendersSeq,
    param::AbstractBendersSeqParam = BendersSeqParam()
)
```

# Examples
```julia
preprocessing = DisjunctiveLPRelaxationPreprocessing(typical_oracle, disj_oracle)
# Use with BendersBnB
env = BendersBnB(master, preprocessing, lazy_callback, user_callback)
```
See also: [`LPRelaxationPreprocessing`](@ref), [`AbstractDisjunctiveOracle`](@ref)
"""
mutable struct DisjunctiveLPRelaxationPreprocessing <: AbstractPreprocessing
    typical_oracle::AbstractTypicalOracle
    disjunctive_oracle::AbstractDisjunctiveOracle
    seq_env_type::Type{<:AbstractBendersSeq}
    param::AbstractBendersSeqParam

    function DisjunctiveLPRelaxationPreprocessing(
        typical_oracle::AbstractTypicalOracle,
        disjunctive_oracle::AbstractDisjunctiveOracle;
        seq_env_type::Type{<:AbstractBendersSeq} = BendersSeq,
        param::AbstractBendersSeqParam = BendersSeqParam()
    )
        new(typical_oracle, disjunctive_oracle, seq_env_type, param)
    end
end

"""
    preprocess!(master::AbstractMaster, preprocessing::DisjunctiveLPRelaxationPreprocessing) -> Float64

Run up to two preprocessing phases using the typical and
disjunctive oracles defined in a `DisjunctiveLPRelaxationPreprocessing` object, and
return the total preprocessing time in seconds.

This routine performs the following steps:

1. **Model relaxation.**  
   All integrality constraints in `master.model` are temporarily relaxed.  
   The original integrality settings are restored automatically upon exit,
   even if an error occurs.

2. **Phase 1 – preprocessing with typical oracle.**  
   A preprocessing is built with `preprocessing.typical_oracle` and
   executed. 

3. **Phase 2 – preprocessing with disjunctive oracle.**  
   A preprocessing follows using `preprocessing.disjunctive_oracle`
   and solved using the remaining time budget.

The function returns the total preprocessing time in seconds.
"""
function preprocess!(master::AbstractMaster, preprocessing::DisjunctiveLPRelaxationPreprocessing; time_limit::Float64 = 100.0)

    tic = time()

    seq_param = deepcopy(preprocessing.param)
    seq_param.time_limit = max(0.0, min(time_limit, seq_param.time_limit))
    
    # Relax integrality, ensure undo() always runs even on error
    undo = relax_integrality(master.model)

    try
        # Phase 1: Preprocessing with typical oracle
        env_preprocessing_typical = preprocessing.seq_env_type(master, preprocessing.typical_oracle; param = seq_param)
        solve!(env_preprocessing_typical)

        remaining_time = max(0.0, seq_param.time_limit - (time() - tic))
        # No time remains for Phase 2: keep the cuts generated so far.
        if remaining_time <= 0.0
            return time() - tic
        end

        # Phase 2: Preprocessing with disjunctive oracle 
        seq_param.time_limit = remaining_time

        env_preprocessing_disjunctive = preprocessing.seq_env_type(master, preprocessing.disjunctive_oracle; param = seq_param)
        solve!(env_preprocessing_disjunctive)
    finally
        # always restore integrality (even on exceptions)
        undo()
    end

    return time() - tic
end
