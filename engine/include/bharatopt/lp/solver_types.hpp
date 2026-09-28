#pragma once
#include "bharatopt/model/types.hpp"
#include <string>

namespace bharatopt {

struct SolverStatistics {
    int iterations = 0;
    double solve_time = 0.0;
    int phase1_iterations = 0;
    int pivots = 0;
};

struct SolverResult {
    TerminationStatus status = TerminationStatus::Unknown;
    double objective_value = 0.0;
    std::vector<double> primal_variables;
    std::vector<double> dual_variables;
    std::vector<double> reduced_costs;
    SolverStatistics stats;
};

class LPSolver {
public:
    virtual ~LPSolver() = default;
    virtual void load_model(class Model* model) = 0;
    virtual SolverResult solve() = 0;
};

} // namespace bharatopt
