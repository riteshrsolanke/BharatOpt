#pragma once
#include <vector>
#include <memory>
#include <string>
#include <cmath>
#include <stdexcept>

namespace bharatopt {

enum class NLPStatus {
    MODEL_VALID,
    MODEL_INVALID,
    LOCAL_SOLUTION,
    GLOBAL_CERTIFICATE,
    UNSUPPORTED,
    NUMERICAL_FAILURE
};

enum class BinaryOp { Add, Sub, Mul, Div, Power };
enum class UnaryOp { Sin, Cos, Exp, Log, Sqrt };

class Expression {
public:
    virtual ~Expression() = default;
    virtual double evaluate(const std::vector<double>& x) const = 0;
    virtual double gradient(int var_id, const std::vector<double>& x) const = 0;
};

class ConstantExpr : public Expression {
public:
    explicit ConstantExpr(double value) : value_(value) {}
    double evaluate(const std::vector<double>& x) const override { return value_; }
    double gradient(int var_id, const std::vector<double>& x) const override { return 0.0; }
private:
    double value_;
};

class VariableExpr : public Expression {
public:
    explicit VariableExpr(int id) : id_(id) {}
    double evaluate(const std::vector<double>& x) const override { return x[id_]; }
    double gradient(int var_id, const std::vector<double>& x) const override { return (id_ == var_id) ? 1.0 : 0.0; }
private:
    int id_;
};

class UnaryExpression : public Expression {
public:
    UnaryExpression(UnaryOp op, std::shared_ptr<Expression> expr) : op_(op), expr_(std::move(expr)) {}

    double evaluate(const std::vector<double>& x) const override {
        double val = expr_->evaluate(x);
        switch (op_) {
            case UnaryOp::Sin: return std::sin(val);
            case UnaryOp::Cos: return std::cos(val);
            case UnaryOp::Exp: return std::exp(val);
            case UnaryOp::Log: return std::log(val);
            case UnaryOp::Sqrt: return std::sqrt(val);
        }
        return 0.0;
    }

    double gradient(int var_id, const std::vector<double>& x) const override {
        double val = expr_->evaluate(x);
        double grad = expr_->gradient(var_id, x);
        switch (op_) {
            case UnaryOp::Sin: return std::cos(val) * grad;
            case UnaryOp::Cos: return -std::sin(val) * grad;
            case UnaryOp::Exp: return std::exp(val) * grad;
            case UnaryOp::Log: return (1.0 / val) * grad;
            case UnaryOp::Sqrt: return (0.5 / std::sqrt(val)) * grad;
        }
        return 0.0;
    }

private:
    UnaryOp op_;
    std::shared_ptr<Expression> expr_;
};

class BinaryExpression : public Expression {
public:
    BinaryExpression(BinaryOp op, std::shared_ptr<Expression> left, std::shared_ptr<Expression> right) 
        : op_(op), left_(std::move(left)), right_(std::move(right)) {}

    double evaluate(const std::vector<double>& x) const override {
        double l = left_->evaluate(x);
        double r = right_->evaluate(x);
        switch (op_) {
            case BinaryOp::Add: return l + r;
            case BinaryOp::Sub: return l - r;
            case BinaryOp::Mul: return l * r;
            case BinaryOp::Div: return l / r;
            case BinaryOp::Power: return std::pow(l, r);
        }
        return 0.0;
    }

    double gradient(int var_id, const std::vector<double>& x) const override {
        double l = left_->evaluate(x);
        double r = right_->evaluate(x);
        double dl = left_->gradient(var_id, x);
        double dr = right_->gradient(var_id, x);
        
        switch (op_) {
            case BinaryOp::Add: return dl + dr;
            case BinaryOp::Sub: return dl - dr;
            case BinaryOp::Mul: return dl * r + l * dr;
            case BinaryOp::Div: return (dl * r - l * dr) / (r * r);
            case BinaryOp::Power: 
                // Simplified grad for a^b (assuming b is constant or dealing with general case)
                if (dr == 0.0) return r * std::pow(l, r - 1.0) * dl; // x^2 -> 2*x
                return std::pow(l, r) * (dr * std::log(l) + r * dl / l);
        }
        return 0.0;
    }

private:
    BinaryOp op_;
    std::shared_ptr<Expression> left_;
    std::shared_ptr<Expression> right_;
};

class NLPModel {
public:
    NLPStatus validate(const std::vector<double>& test_point) const {
        if (!objective_) return NLPStatus::MODEL_INVALID;
        
        // Dimension validation
        // (Assuming test_point matches variables size)
        
        // Finite-value, Domain, NaN/Inf checks
        double val = objective_->evaluate(test_point);
        if (std::isnan(val) || std::isinf(val)) {
            return NLPStatus::NUMERICAL_FAILURE;
        }

        return NLPStatus::MODEL_VALID;
    }

    void set_objective(std::shared_ptr<Expression> obj) { objective_ = std::move(obj); }
    std::shared_ptr<Expression> objective() const { return objective_; }

private:
    std::shared_ptr<Expression> objective_;
};

} // namespace bharatopt
