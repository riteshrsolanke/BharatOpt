#pragma once
#include "types.hpp"
#include <string>
#include <vector>
#include <utility>

namespace bharatopt {

class Constraint {
public:
    Constraint(int id, std::string name, ConstraintType type, double rhs)
        : id_(id), name_(std::move(name)), type_(type), rhs_(rhs) {}

    void add_term(int var_id, double coefficient) {
        terms_.emplace_back(var_id, coefficient);
    }

    int id() const { return id_; }
    const std::string& name() const { return name_; }
    ConstraintType type() const { return type_; }
    double rhs() const { return rhs_; }
    const std::vector<std::pair<int, double>>& terms() const { return terms_; }

private:
    int id_;
    std::string name_;
    ConstraintType type_;
    double rhs_;
    std::vector<std::pair<int, double>> terms_;
};

} // namespace bharatopt
