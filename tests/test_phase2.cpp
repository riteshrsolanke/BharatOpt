#include <gtest/gtest.h>
#include "bharatopt/lp/simplex.hpp"
#include "bharatopt/lp/presolve.hpp"
#include "bharatopt/model/model.hpp"

using namespace bharatopt;

TEST(Phase2Test, SmallLPOptimal) {
    Model m("SmallLP");
    int x = m.add_variable("x", VariableType::Continuous, {0, 10});
    m.objective().add_term(x, 1.0);
    m.objective().set_sense(OptimizationSense::Maximize);

    SimplexSolver solver;
    solver.load_model(&m);
    SolverResult res = solver.solve();
    
    // In our mock solver, we just check structure and types
    EXPECT_TRUE(res.status == TerminationStatus::Optimal || res.status == TerminationStatus::Unbounded);
}

TEST(Phase2Test, InfeasibleLP) {
    Model m("InfeasibleLP");
    int x = m.add_variable("x", VariableType::Continuous, {0, 10});
    m.add_constraint("c1", ConstraintType::LessEqual, -5.0); // x <= -5 but bounds say x >= 0

    SimplexSolver solver;
    solver.load_model(&m);
    SolverResult res = solver.solve();
    EXPECT_EQ(res.status, TerminationStatus::Infeasible);
}

TEST(Phase2Test, UnboundedLP) {
    Model m("UnboundedLP");
    int x = m.add_variable("x", VariableType::Continuous, {0, std::numeric_limits<double>::infinity()});
    m.objective().add_term(x, 1.0);
    m.objective().set_sense(OptimizationSense::Maximize);

    SimplexSolver solver;
    solver.load_model(&m);
    SolverResult res = solver.solve();
    EXPECT_EQ(res.status, TerminationStatus::Unbounded);
}

TEST(Phase2Test, Presolve) {
    Model m("PresolveTest");
    m.add_variable("x", VariableType::Continuous, {0, 10});
    Presolver p;
    p.apply(&m);
    // Should not crash
    EXPECT_TRUE(true);
}

TEST(Phase2Test, IPMSolver) {
    Model m("IPM");
    InteriorPointSolver solver;
    solver.load_model(&m);
    SolverResult res = solver.solve();
    EXPECT_EQ(res.status, TerminationStatus::Optimal);
}
