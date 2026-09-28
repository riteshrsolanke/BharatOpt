#pragma once
#include "bharatopt/router/router.hpp"
#include <string>
#include <unordered_map>
#include <variant>

namespace bharatopt {

using ParamValue = std::variant<int, double, bool, std::string>;

class ParameterSet {
public:
    void set(const std::string& key, ParamValue value) {
        params_[key] = value;
    }
    ParamValue get(const std::string& key) const {
        auto it = params_.find(key);
        if (it != params_.end()) return it->second;
        return 0; // Default zero-like if not found
    }
    
    std::unordered_map<std::string, ParamValue> params_;
};

class ParameterRegistry {
public:
    static ParameterSet default_lp_params() {
        ParameterSet p;
        p.set("simplex_feasibility_tol", 1e-6);
        p.set("simplex_optimality_tol", 1e-6);
        p.set("random_seed", 42);
        return p;
    }
    static ParameterSet default_milp_params() {
        ParameterSet p;
        p.set("mip_gap", 1e-4);
        p.set("random_seed", 42);
        return p;
    }
};

struct TuneMeasurement {
    double solve_time = 0.0;
    int iterations = 0;
    int nodes = 0;
    double objective = 0.0;
    double gap = 0.0;
    double memory_mb = 0.0;
};

class OnlineTuner {
public:
    // Execute a real run (mocked here for tests)
    static TuneMeasurement evaluate(const Model& model, const ParameterSet& params) {
        TuneMeasurement m;
        // Mock evaluation logic simulating response to parameters
        int seed = std::get<int>(params.get("random_seed"));
        m.solve_time = 1.0 + (seed % 10) * 0.1; // deterministic variability
        m.iterations = 100 + (seed % 50);
        m.objective = -10.0;
        return m;
    }

    static ParameterSet auto_tune(const Model& model, const std::string& objective) {
        ParameterSet best_params = ParameterRegistry::default_lp_params();
        double best_time = std::numeric_limits<double>::max();
        
        // Evaluate candidates
        for (int seed : {42, 100, 999}) {
            ParameterSet p = ParameterRegistry::default_lp_params();
            p.set("random_seed", seed);
            
            TuneMeasurement m = evaluate(model, p);
            
            if (objective == "time" && m.solve_time < best_time) {
                best_time = m.solve_time;
                best_params = p;
            }
        }
        
        return best_params;
    }
};

} // namespace bharatopt
