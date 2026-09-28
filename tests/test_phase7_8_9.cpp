#include <gtest/gtest.h>
#include "bharatopt/model/model.hpp"
#include "bharatopt/router/router.hpp"
#include "bharatopt/tuning/tuning.hpp"
#include "bharatopt/backend/backend.hpp"

using namespace bharatopt;

TEST(Phase7Test, SolverRouter_LP) {
    Model m("LPModel");
    m.add_variable("x", VariableType::Continuous, {0, 10});
    
    RoutingResult result = SolverRouter::route(m);
    
    EXPECT_EQ(result.problem_type, ProblemType::LP);
    EXPECT_EQ(result.selected_solver, "LP Engine");
    EXPECT_EQ(result.selected_backend, "CPU");
}

TEST(Phase7Test, SolverRouter_MILP) {
    Model m("MILPModel");
    m.add_variable("x", VariableType::Integer, {0, 10});
    m.add_constraint("c1", ConstraintType::LessEqual, 5.0);
    
    RoutingResult result = SolverRouter::route(m);
    
    EXPECT_EQ(result.problem_type, ProblemType::MILP);
    EXPECT_EQ(result.selected_solver, "MILP Engine");
    EXPECT_EQ(result.selected_algorithm, "Branch-and-Bound + LP Relaxation (Revised Simplex)");
}

TEST(Phase7Test, SolverRouter_QP) {
    Model m("QPModel");
    int x = m.add_variable("x", VariableType::Continuous, {0, 10});
    m.objective().add_quadratic_term(x, x, 1.0);
    
    RoutingResult result = SolverRouter::route(m);
    
    EXPECT_EQ(result.problem_type, ProblemType::QP);
    EXPECT_EQ(result.selected_solver, "QP Engine");
}

TEST(Phase8Test, ParameterTuning) {
    Model m("TuneModel");
    ParameterSet best = OnlineTuner::auto_tune(m, "time");
    
    // Seed 42 gives solve time 1.2, seed 100 gives 1.0 (since 100%10 = 0 -> 1.0), seed 999 gives 1.9
    // So 100 should be the best seed
    EXPECT_EQ(std::get<int>(best.get("random_seed")), 100);
}

TEST(Phase9Test, BackendSelection) {
    CPUBackend cpu;
    EXPECT_TRUE(cpu.is_available());
    
    CUDABackend gpu;
    // GPU should be false on this CI machine if USE_CUDA is not defined
#ifndef USE_CUDA
    EXPECT_FALSE(gpu.is_available());
#endif

    std::vector<double> values = {1.0, 2.0};
    std::vector<int> col_indices = {0, 1};
    std::vector<int> row_ptr = {0, 2};
    std::vector<double> x = {2.0, 3.0};
    
    std::vector<double> y = cpu.spmv(values, col_indices, row_ptr, x);
    EXPECT_EQ(y.size(), 1);
    EXPECT_DOUBLE_EQ(y[0], 1.0*2.0 + 2.0*3.0); // 8.0
    
    BackendResult res = cpu.last_result();
    EXPECT_GT(res.cpu_time, 0.0);
}
