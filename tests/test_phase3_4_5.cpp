#include <gtest/gtest.h>
#include "bharatopt/model/model.hpp"
#include "bharatopt/lp/iteration_record.hpp"
#include "bharatopt/milp/solver.hpp"
#include "bharatopt/api/json_exporter.hpp"
#include <vector>

using namespace bharatopt;

TEST(Phase345Test, DataGeneration) {
    Model m("MILP_Test");
    m.add_variable("x", VariableType::Integer, {0, 10});
    m.add_variable("y", VariableType::Integer, {0, 10});
    m.objective().set_sense(OptimizationSense::Maximize);

    // Mock LP Iterations
    std::vector<LPIterationRecord> lp_iters;
    lp_iters.push_back({1, 0, 1, 1.5, 0.0, {0, 2}, {-0.5, -0.2}, {0.0, 0.0}});
    lp_iters.push_back({2, 1, 2, -2.0, 5.0, {0, 1}, {0.0, 0.0}, {1.5, 2.0}});

    // Run MILP Solver
    MILPSolver solver;
    solver.load_model(&m);
    solver.add_cut_generator(std::make_unique<GomoryCutGenerator>());
    
    MILPResult res = solver.solve();
    
    EXPECT_GT(res.tree_nodes.size(), 0);
    EXPECT_EQ(res.stats.nodes_explored, 3);
    EXPECT_EQ(res.stats.lp_relaxations_solved, 3);

    // Export to JSON for the Web UI
    JSONExporter::export_data("../web/data.json", lp_iters, res.tree_nodes);
}
