#include <gtest/gtest.h>
#include "bharatopt/nlp/nlp.hpp"
#include "bharatopt/cert/certificate.hpp"
#include "bharatopt/uncertainty/uncertainty.hpp"

using namespace bharatopt;

TEST(Phase10Test, NLPExpressionDAG) {
    // f(x) = sin(x) + x^2
    auto var_x = std::make_shared<VariableExpr>(0); // x is at index 0
    auto sin_x = std::make_shared<UnaryExpression>(UnaryOp::Sin, var_x);
    auto const_2 = std::make_shared<ConstantExpr>(2.0);
    auto x_sq = std::make_shared<BinaryExpression>(BinaryOp::Power, var_x, const_2);
    auto func = std::make_shared<BinaryExpression>(BinaryOp::Add, sin_x, x_sq);
    
    NLPModel m;
    m.set_objective(func);
    
    std::vector<double> point = { 1.5 }; // x = 1.5
    EXPECT_EQ(m.validate(point), NLPStatus::MODEL_VALID);
    
    double val = func->evaluate(point);
    // sin(1.5) + 1.5^2 = 0.997495 + 2.25 = 3.247495
    EXPECT_NEAR(val, std::sin(1.5) + 2.25, 1e-5);
    
    // Gradient wrt x: cos(x) + 2x = cos(1.5) + 3.0
    double grad = func->gradient(0, point);
    EXPECT_NEAR(grad, std::cos(1.5) + 3.0, 1e-5);
}

TEST(Phase11Test, LPCertification) {
    PrimalCertificate primal = {{1.0, 2.0}, 100.0, {0.0, 0.0}};
    DualCertificate dual = {{5.0, 5.0}, 100.0, {0.0, 0.0}};
    LPTolerances tol;
    
    CertificationStatus status = CertificateVerifier::verify_lp(primal, dual, tol, "hash_123", "v1.0");
    EXPECT_EQ(status, CertificationStatus::CERTIFIED);
    
    // Invalid duality gap
    DualCertificate invalid_dual = {{5.0, 5.0}, 99.0, {0.0, 0.0}};
    EXPECT_EQ(CertificateVerifier::verify_lp(primal, invalid_dual, tol, "hash", "v1.0"), CertificationStatus::INVALID_PROOF);
}

TEST(Phase11Test, MILPProofLogging) {
    MILPProofLog proof;
    proof.nodes.push_back({0, 10.0, "INFEASIBLE_LP"});
    proof.nodes.push_back({1, 15.0, "BOUND_PRUNING"});
    proof.terminal_state = "OPTIMAL";
    
    EXPECT_EQ(MIPProofVerifier::verify_mip(proof), CertificationStatus::CERTIFIED);
}

TEST(Phase12Test, UncertaintyModeling) {
    UncertaintyModel model;
    
    // Refinery sulfur feed example
    UncertainParameter feed_sulfur;
    feed_sulfur.name = "feed_sulfur";
    feed_sulfur.nominal = 2.4;
    feed_sulfur.range = {2.1, 2.8};
    model.add_uncertain_parameter(feed_sulfur);
    
    ScenarioSet scenarios;
    scenarios.scenarios.push_back({0.5, {{"feed_sulfur", 2.1}}});
    scenarios.scenarios.push_back({0.5, {{"feed_sulfur", 2.8}}});
    model.set_scenario_set(scenarios);
    
    UncertaintyResult res = model.solve_robust();
    EXPECT_GT(res.nominal_solution.size(), 0);
    EXPECT_GT(res.robust_solution.size(), 0);
    EXPECT_EQ(res.scenario_results.size(), 2);
    // Robust solution should typically be more conservative than nominal
    EXPECT_LT(res.robust_solution[0], res.nominal_solution[0]);
}
