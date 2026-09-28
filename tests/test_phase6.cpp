#include <gtest/gtest.h>
#include "bharatopt/model/model.hpp"
#include "bharatopt/qp/solver.hpp"
#include "bharatopt/qp/admm.hpp"

using namespace bharatopt;

TEST(Phase6Test, QPObjective) {
    Model m("QPTest");
    int x = m.add_variable("x", VariableType::Continuous, {0, 10});
    // min 0.5 * x^2 - x
    m.objective().set_sense(OptimizationSense::Minimize);
    m.objective().add_quadratic_term(x, x, 0.5);
    m.objective().add_term(x, -1.0);
    
    EXPECT_EQ(m.objective().quadratic_terms().size(), 1);
    
    auto [v1, v2, coeff] = m.objective().quadratic_terms()[0];
    EXPECT_EQ(v1, x);
    EXPECT_EQ(v2, x);
    EXPECT_DOUBLE_EQ(coeff, 0.5);
}

TEST(Phase6Test, ActiveSetQPSolver) {
    Model m("QPTest");
    m.add_variable("x", VariableType::Continuous, {0, 10});
    ActiveSetQPSolver solver;
    solver.load_model(&m);
    SolverResult res = solver.solve();
    
    EXPECT_EQ(res.status, TerminationStatus::Optimal);
    EXPECT_DOUBLE_EQ(res.primal_variables[0], 1.0);
    EXPECT_DOUBLE_EQ(res.objective_value, -0.5);
    EXPECT_TRUE(solver.check_feasibility());
    EXPECT_TRUE(solver.check_convergence());
}

TEST(Phase6Test, InteriorPointQPSolver) {
    Model m("QPTest");
    m.add_variable("x", VariableType::Continuous, {0, 10});
    InteriorPointQPSolver solver;
    solver.load_model(&m);
    SolverResult res = solver.solve();
    
    EXPECT_EQ(res.status, TerminationStatus::Optimal);
    EXPECT_DOUBLE_EQ(res.primal_variables[0], 1.0);
    EXPECT_DOUBLE_EQ(res.objective_value, -0.5);
}

TEST(Phase6Test, ADMMFramework) {
    Model m("ADMMTest");
    m.add_variable("x", VariableType::Continuous, {0, 10});
    ADMMProblem prob(&m);
    
    ADMMParameters params;
    params.rho = 2.0;
    ADMMSolver solver(params);
    
    ADMMResult res = solver.solve(prob);
    
    EXPECT_EQ(res.status, TerminationStatus::Optimal);
    EXPECT_GT(res.stats.iterations, 0);
    EXPECT_EQ(res.rho_history[0], 2.0);
    EXPECT_GT(res.primal_residuals.size(), 0);
    EXPECT_GT(res.dual_residuals.size(), 0);
    
    // Objective tracking tests
    EXPECT_DOUBLE_EQ(res.primal_variables[0], 1.0);
    EXPECT_DOUBLE_EQ(res.objective_value, -0.5);
}
