#pragma once
#include <vector>
#include <string>
#include <cmath>
#include <functional>
#include "bharatopt/model/model.hpp"

namespace bharatopt {

enum class SensitivityStatus {
    AVAILABLE,
    GRADIENT_UNAVAILABLE
};

struct GradientResult {
    std::vector<double> solution;
    std::vector<double> gradient;
    std::vector<std::vector<double>> jacobian;
    SensitivityStatus status = SensitivityStatus::GRADIENT_UNAVAILABLE;
};

struct LPSensitivity {
    std::vector<double> dual_variables;
    std::vector<double> shadow_prices;
    std::vector<double> reduced_costs;
    bool valid = false;
};

class DifferentiableProblem {
public:
    virtual ~DifferentiableProblem() = default;
    virtual double evaluate(const std::vector<double>& params) = 0;
    // Differentiable through valid KKT conditions
    virtual GradientResult differentiate(const std::vector<double>& params) = 0;
};

class OptimizationLayer {
public:
    static GradientResult forward_and_backward(DifferentiableProblem& prob, const std::vector<double>& params) {
        return prob.differentiate(params);
    }
};

class SensitivityAnalyzer {
public:
    static LPSensitivity analyze_lp(Model* model, const std::vector<double>& primal, const std::vector<double>& dual) {
        LPSensitivity s;
        if (dual.empty()) return s;
        
        s.dual_variables = dual;
        s.shadow_prices = dual; // For simple LPs, shadow price = dual
        
        // Mock reduced costs calculation
        s.reduced_costs.resize(primal.size(), 0.0);
        s.valid = true;
        return s;
    }
    
    // Utility for finite-difference verification
    static std::vector<double> finite_difference(
        const std::vector<double>& params, 
        std::function<double(const std::vector<double>&)> eval_func, 
        double h = 1e-5) 
    {
        std::vector<double> grad(params.size(), 0.0);
        std::vector<double> p_plus = params;
        std::vector<double> p_minus = params;
        
        for (size_t i = 0; i < params.size(); ++i) {
            p_plus[i] += h;
            p_minus[i] -= h;
            grad[i] = (eval_func(p_plus) - eval_func(p_minus)) / (2 * h);
            p_plus[i] = params[i];
            p_minus[i] = params[i];
        }
        return grad;
    }
};

} // namespace bharatopt
