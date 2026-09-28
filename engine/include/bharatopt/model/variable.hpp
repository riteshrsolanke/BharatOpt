#pragma once
#include "types.hpp"
#include <string>
#include <limits>

namespace bharatopt {

struct Bounds {
    double lower = 0.0;
    double upper = std::numeric_limits<double>::infinity();
};

class Variable {
public:
    Variable(int id, std::string name, VariableType type, Bounds bounds)
        : id_(id), name_(std::move(name)), type_(type), bounds_(bounds) {}

    int id() const { return id_; }
    const std::string& name() const { return name_; }
    VariableType type() const { return type_; }
    const Bounds& bounds() const { return bounds_; }

private:
    int id_;
    std::string name_;
    VariableType type_;
    Bounds bounds_;
};

} // namespace bharatopt
