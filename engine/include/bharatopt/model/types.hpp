#pragma once
#include <string>

namespace bharatopt {

enum class VariableType {
    Continuous,
    Integer,
    Binary
};

enum class ConstraintType {
    LessEqual,
    GreaterEqual,
    Equal
};

enum class OptimizationSense {
    Minimize,
    Maximize
};

enum class TerminationStatus {
    Optimal,
    Infeasible,
    Unbounded,
    IterationLimit,
    TimeLimit,
    NumericalError,
    Unknown
};

} // namespace bharatopt
