#pragma once
#include "bharatopt/model/model.hpp"
#include <string>

namespace bharatopt {

enum class ProblemType { LP, MILP, QP, MIQP, NLP, Uncertain, Unsupported };

struct InstanceProfile {
    int variables = 0;
    int constraints = 0;
    int nonzeros = 0;
    double density = 0.0;
    double integer_ratio = 0.0;
    double binary_ratio = 0.0;
    double coefficient_range_min = 0.0;
    double coefficient_range_max = 0.0;
    double row_density = 0.0;
    double column_density = 0.0;
    bool has_quadratic = false;
    bool has_nonlinear = false;
    bool is_uncertain = false;
};

class ModelAnalyzer {
public:
    static InstanceProfile analyze(const Model& model) {
        InstanceProfile profile;
        profile.variables = static_cast<int>(model.variables().size());
        profile.constraints = static_cast<int>(model.constraints().size());
        
        int ints = 0, bins = 0;
        for (const auto& v : model.variables()) {
            if (v->type() == VariableType::Integer) ints++;
            if (v->type() == VariableType::Binary) bins++;
        }
        
        if (profile.variables > 0) {
            profile.integer_ratio = static_cast<double>(ints) / profile.variables;
            profile.binary_ratio = static_cast<double>(bins) / profile.variables;
        }

        profile.has_quadratic = !model.objective().quadratic_terms().empty();
        
        // Count nonzeros in linear constraints
        for (const auto& c : model.constraints()) {
            profile.nonzeros += static_cast<int>(c->terms().size());
        }
        
        if (profile.variables > 0 && profile.constraints > 0) {
            profile.density = static_cast<double>(profile.nonzeros) / 
                              (profile.variables * profile.constraints);
        }
        
        return profile;
    }
};

class ProblemClassifier {
public:
    static ProblemType classify(const InstanceProfile& profile) {
        if (profile.is_uncertain) return ProblemType::Uncertain;
        if (profile.has_nonlinear) return ProblemType::NLP;
        
        bool has_integers = (profile.integer_ratio > 0 || profile.binary_ratio > 0);
        
        if (profile.has_quadratic) {
            return has_integers ? ProblemType::MIQP : ProblemType::QP;
        } else {
            return has_integers ? ProblemType::MILP : ProblemType::LP;
        }
    }
};

struct RoutingResult {
    ProblemType problem_type;
    std::string selected_solver;
    std::string selected_algorithm;
    std::string selected_backend;
    std::string reason;
    std::string fallback;
    double confidence = 0.0;
};

class BackendRouter {
public:
    static std::string select_backend(const InstanceProfile& profile) {
        // Simple heuristic: choose GPU if model is dense/large, else CPU
        if (profile.variables > 10000 && profile.density > 0.05) {
            return "GPU";
        }
        return "CPU";
    }
};

class SolverRouter {
public:
    static RoutingResult route(const Model& model) {
        RoutingResult result;
        InstanceProfile profile = ModelAnalyzer::analyze(model);
        result.problem_type = ProblemClassifier::classify(profile);
        result.selected_backend = BackendRouter::select_backend(profile);
        
        switch (result.problem_type) {
            case ProblemType::MILP:
                result.selected_solver = "MILP Engine";
                result.selected_algorithm = "Branch-and-Bound + LP Relaxation (Revised Simplex)";
                result.reason = "The model contains integer variables and linear constraints.";
                result.fallback = "Branch-and-Bound + Interior Point";
                result.confidence = 0.95;
                break;
            case ProblemType::LP:
                result.selected_solver = "LP Engine";
                result.selected_algorithm = profile.density < 0.1 ? "Revised Simplex" : "Interior Point Method";
                result.reason = "The model is continuous and linear.";
                result.fallback = profile.density < 0.1 ? "Interior Point Method" : "Revised Simplex";
                result.confidence = 0.99;
                break;
            case ProblemType::QP:
                result.selected_solver = "QP Engine";
                result.selected_algorithm = "Primal-Dual Interior Point QP";
                result.reason = "The model contains quadratic objective terms and continuous variables.";
                result.fallback = "Active Set QP";
                result.confidence = 0.92;
                break;
            default:
                result.selected_solver = "Unknown";
                result.selected_algorithm = "None";
                result.reason = "Unsupported model type structure detected.";
                result.fallback = "None";
                result.confidence = 0.0;
        }
        
        return result;
    }
};

} // namespace bharatopt
