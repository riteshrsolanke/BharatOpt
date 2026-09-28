#pragma once
#include <string>
#include <vector>
#include <map>
#include <memory>

namespace bharatopt {

struct IntervalSet {
    double lower;
    double upper;
};

struct UncertainParameter {
    std::string name;
    double nominal;
    IntervalSet range;
};

struct Scenario {
    double probability;
    std::map<std::string, double> realizations;
};

struct ScenarioSet {
    std::vector<Scenario> scenarios;
};

struct RobustConstraint {
    std::string name;
    // Logical representation of an uncertain constraint mapping
};

struct UncertaintyResult {
    std::vector<double> nominal_solution;
    std::vector<double> robust_solution;
    std::vector<std::vector<double>> scenario_results;
    double worst_case_objective = 0.0;
    std::vector<double> constraint_margins;
};

class UncertaintyModel {
public:
    void add_uncertain_parameter(const UncertainParameter& param) {
        parameters_.push_back(param);
    }

    void set_scenario_set(const ScenarioSet& set) {
        scenario_set_ = set;
    }

    UncertaintyResult solve_robust() {
        // Automatically formulate robust counterpart / scenario model (mocked)
        UncertaintyResult res;
        res.nominal_solution = {100.0};
        res.robust_solution = {90.0}; // Backed off due to uncertainty
        res.worst_case_objective = 85.0;
        res.constraint_margins = {5.0};
        
        if (!scenario_set_.scenarios.empty()) {
            res.scenario_results.push_back({95.0});
            res.scenario_results.push_back({85.0});
        }
        return res;
    }

private:
    std::vector<UncertainParameter> parameters_;
    ScenarioSet scenario_set_;
};

} // namespace bharatopt
