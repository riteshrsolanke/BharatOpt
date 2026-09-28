#pragma once
#include "types.hpp"
#include <vector>
#include <utility>

namespace bharatopt {

class Objective {
public:
    explicit Objective(OptimizationSense sense = OptimizationSense::Minimize) 
        : sense_(sense), offset_(0.0) {}

    void set_sense(OptimizationSense sense) { sense_ = sense; }
    OptimizationSense sense() const { return sense_; }

    void add_term(int var_id, double coefficient) {
        linear_terms_.emplace_back(var_id, coefficient);
    }
    
    void add_quadratic_term(int var_id1, int var_id2, double coefficient) {
        quadratic_terms_.emplace_back(var_id1, var_id2, coefficient);
    }

    void set_offset(double offset) { offset_ = offset; }
    double offset() const { return offset_; }

    const std::vector<std::pair<int, double>>& linear_terms() const { return linear_terms_; }
    const std::vector<std::tuple<int, int, double>>& quadratic_terms() const { return quadratic_terms_; }

private:
    OptimizationSense sense_;
    double offset_;
    std::vector<std::pair<int, double>> linear_terms_;
    std::vector<std::tuple<int, int, double>> quadratic_terms_;
};

} // namespace bharatopt
