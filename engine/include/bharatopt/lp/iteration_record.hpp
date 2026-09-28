#pragma once
#include <vector>

namespace bharatopt {

struct LPIterationRecord {
    int iteration;
    int entering_variable;
    int leaving_variable;
    double pivot_element;
    double objective_value;
    std::vector<int> current_basis;
    std::vector<double> reduced_costs;
    std::vector<double> current_solution;
};

} // namespace bharatopt
