#pragma once
#include "variable.hpp"
#include "constraint.hpp"
#include "objective.hpp"
#include <memory>
#include <string>

namespace bharatopt {

class Model {
public:
    Model(std::string name = "BharatOptModel") : name_(std::move(name)) {}

    int add_variable(const std::string& name, VariableType type, Bounds bounds) {
        int id = static_cast<int>(variables_.size());
        variables_.emplace_back(std::make_unique<Variable>(id, name, type, bounds));
        return id;
    }

    int add_constraint(const std::string& name, ConstraintType type, double rhs) {
        int id = static_cast<int>(constraints_.size());
        constraints_.emplace_back(std::make_unique<Constraint>(id, name, type, rhs));
        return id;
    }

    Constraint* get_constraint(int id) { return constraints_[id].get(); }
    Variable* get_variable(int id) { return variables_[id].get(); }
    const std::vector<std::unique_ptr<Variable>>& variables() const { return variables_; }
    const std::vector<std::unique_ptr<Constraint>>& constraints() const { return constraints_; }

    Objective& objective() { return objective_; }
    const Objective& objective() const { return objective_; }

private:
    std::string name_;
    std::vector<std::unique_ptr<Variable>> variables_;
    std::vector<std::unique_ptr<Constraint>> constraints_;
    Objective objective_;
};

} // namespace bharatopt
