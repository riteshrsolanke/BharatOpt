#include <gtest/gtest.h>
#include "bharatopt/differentiable/differentiable.hpp"
#include "bharatopt/ml/gnn.hpp"
#include "bharatopt/model/model.hpp"

using namespace bharatopt;

TEST(Phase13Test, SensitivityFiniteDifference) {
    // f(x) = x^2
    auto func = [](const std::vector<double>& params) {
        return params[0] * params[0];
    };
    
    std::vector<double> x = {2.0};
    std::vector<double> grad = SensitivityAnalyzer::finite_difference(x, func);
    
    // Analytical gradient of x^2 at x=2 is 2*x = 4.0
    EXPECT_EQ(grad.size(), 1);
    EXPECT_NEAR(grad[0], 4.0, 1e-4);
    
    Model m;
    LPSensitivity sens = SensitivityAnalyzer::analyze_lp(&m, {1.0}, {0.5});
    EXPECT_TRUE(sens.valid);
    EXPECT_EQ(sens.dual_variables[0], 0.5);
}

TEST(Phase14Test, GNNBranchingFallback) {
    Model m("GNNTest");
    m.add_variable("x", VariableType::Integer, {0, 10});
    
    GNNBranchingModel gnn_model;
    GNNLog log;
    
    int branch_var = gnn_model.select_branching_variable(m, log);
    
    // Because the C++ side doesn't have an active IPC connection to the python sidecar in this test,
    // it MUST trigger the mathematical correctness fallback.
    EXPECT_FALSE(log.gnn_used);
    EXPECT_EQ(log.fallback_reason, "inference_failure_or_timeout");
    EXPECT_EQ(branch_var, 0); // Fell back to classical variable 0
}
