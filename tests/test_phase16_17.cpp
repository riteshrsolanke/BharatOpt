#include <gtest/gtest.h>
#include "bharatopt/model/model.hpp"
#include "../../industrial/refinery/refinery_workflow.hpp"

// End-to-end includes
#include "bharatopt/router/router.hpp"
#include "bharatopt/tuning/tuning.hpp"
#include "bharatopt/cert/certificate.hpp"
#include "bharatopt/backend/backend.hpp"
#include "bharatopt/ml/gnn.hpp"
#include "bharatopt/differentiable/differentiable.hpp"
#include "bharatopt/uncertainty/uncertainty.hpp"

using namespace bharatopt;
using namespace bharatopt::refinery;

TEST(Phase16Test, RefineryOptimizationWorkflow) {
    std::vector<Feedstock> feeds = { {"CrudeA", 1000.0, 50.0, 1.5}, {"CrudeB", 500.0, 60.0, 0.5} };
    std::vector<Product> products = { {"Gasoline", 800.0, 120.0, 0.01}, {"Diesel", 600.0, 110.0, 0.05} };
    std::vector<UnitOperation> units = { {"Distillation", 2000.0, 5.0, 2.0, {0.4, 0.4, 0.2}} };
    
    Model m = RefineryModelBuilder::build(feeds, products, units);
    EconomicAnalysis analysis = RefineryModelBuilder::run_workflow(m);
    
    // Validate output logic
    EXPECT_EQ(analysis.revenue, 1500000.0);
    EXPECT_EQ(analysis.estimated_margin, 700000.0);
    EXPECT_GT(analysis.binding_constraints.size(), 0);
    
    // Check binding constraint exposition
    EXPECT_EQ(analysis.binding_constraints[0].first, "FeedAvailability_CrudeA");
    EXPECT_EQ(analysis.binding_constraints[0].second, 12.50); // Shadow price impact
}

TEST(Phase17Test, FullSystemValidation) {
    // 1. LP / MILP / Router / Tuning Validation
    Model m("SystemValidation");
    m.add_variable("x", VariableType::Integer, {0, 10});
    m.add_constraint("c1", ConstraintType::LessEqual, 5.0);
    
    RoutingResult router_res = SolverRouter::route(m);
    EXPECT_EQ(router_res.problem_type, ProblemType::MILP);
    
    ParameterSet best = OnlineTuner::auto_tune(m, "time");
    EXPECT_EQ(std::get<int>(best.get("random_seed")), 100);
    
    // 2. Certificate Verification
    PrimalCertificate primal = {{1.0}, 10.0, {0.0}};
    DualCertificate dual = {{1.0}, 10.0, {0.0}};
    LPTolerances tol;
    EXPECT_EQ(CertificateVerifier::verify_lp(primal, dual, tol, "hash", "v1.0"), CertificationStatus::CERTIFIED);
    
    // 3. Backend Fallback Verification
    CPUBackend cpu;
    EXPECT_TRUE(cpu.is_available());
    
    // 4. ML / GNN Verification
    GNNBranchingModel gnn;
    GNNLog gnn_log;
    int var = gnn.select_branching_variable(m, gnn_log);
    EXPECT_FALSE(gnn_log.gnn_used);
    
    // 5. Uncertainty Verification
    UncertaintyModel u_model;
    u_model.add_uncertain_parameter({"param1", 10.0, {9.0, 11.0}});
    UncertaintyResult u_res = u_model.solve_robust();
    EXPECT_GT(u_res.nominal_solution.size(), 0);
}
