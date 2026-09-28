#include <gtest/gtest.h>
#include "bharatopt/model/model.hpp"
#include "bharatopt/linalg/sparse.hpp"
#include "bharatopt/linalg/numerical.hpp"

using namespace bharatopt;

TEST(Phase1Test, ModelConstruction) {
    Model m("TestModel");
    int v1 = m.add_variable("x", VariableType::Continuous, {0, 10});
    int v2 = m.add_variable("y", VariableType::Integer, {0, 5});
    
    EXPECT_EQ(m.variables().size(), 2);
    EXPECT_EQ(m.get_variable(v1)->name(), "x");
    EXPECT_EQ(m.get_variable(v2)->type(), VariableType::Integer);

    int c1 = m.add_constraint("c1", ConstraintType::LessEqual, 10.0);
    m.get_constraint(c1)->add_term(v1, 2.0);
    m.get_constraint(c1)->add_term(v2, 1.0);

    EXPECT_EQ(m.constraints().size(), 1);
    EXPECT_EQ(m.get_constraint(c1)->rhs(), 10.0);
    
    m.objective().set_sense(OptimizationSense::Maximize);
    m.objective().add_term(v1, 1.0);
    m.objective().add_term(v2, 2.0);
    
    EXPECT_EQ(m.objective().sense(), OptimizationSense::Maximize);
}

TEST(Phase1Test, SparseMatrix) {
    COOMatrix coo(3, 3);
    coo.add_value(0, 0, 1.0);
    coo.add_value(0, 2, 2.0);
    coo.add_value(1, 1, 3.0);
    coo.add_value(2, 0, 4.0);
    coo.add_value(2, 2, 5.0);

    EXPECT_EQ(coo.nnz(), 5);
    EXPECT_EQ(coo.stats().nnz, 5);

    CSRMatrix csr(coo);
    EXPECT_EQ(csr.nnz(), 5);
    
    std::vector<double> x = {1.0, 2.0, 3.0};
    std::vector<double> y = csr.spmv(x);
    // Row 0: 1*1 + 2*3 = 7
    // Row 1: 3*2 = 6
    // Row 2: 4*1 + 5*3 = 19
    EXPECT_DOUBLE_EQ(y[0], 7.0);
    EXPECT_DOUBLE_EQ(y[1], 6.0);
    EXPECT_DOUBLE_EQ(y[2], 19.0);

    CSCMatrix csc(coo);
    EXPECT_EQ(csc.nnz(), 5);
    CSCMatrix csc_t = csc.transpose();
    EXPECT_EQ(csc_t.nnz(), 5);
    EXPECT_EQ(csc_t.rows(), 3);
}

TEST(Phase1Test, NumericalUtils) {
    std::vector<double> v = {-1.5, 2.0, 0.5};
    EXPECT_DOUBLE_EQ(NumericalUtils::infinity_norm(v), 2.0);

    std::vector<double> Ax = {1.0, 2.0};
    std::vector<double> b = {1.1, 1.9};
    std::vector<double> r = NumericalUtils::compute_residual(Ax, b);
    EXPECT_NEAR(r[0], 0.1, 1e-9);
    EXPECT_NEAR(r[1], -0.1, 1e-9);

    EXPECT_TRUE(NumericalUtils::is_finite(v));
    std::vector<double> v_inf = {1.0, std::numeric_limits<double>::infinity()};
    EXPECT_FALSE(NumericalUtils::is_finite(v_inf));
}
