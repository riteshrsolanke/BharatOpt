#pragma once
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace bharatopt {

class NumericalUtils {
public:
    static double infinity_norm(const std::vector<double>& vec) {
        double norm = 0.0;
        for (double v : vec) {
            norm = std::max(norm, std::abs(v));
        }
        return norm;
    }

    static std::vector<double> compute_residual(const std::vector<double>& Ax, const std::vector<double>& b) {
        if (Ax.size() != b.size()) throw std::invalid_argument("Size mismatch in residual");
        std::vector<double> r(b.size());
        for (size_t i = 0; i < b.size(); ++i) {
            r[i] = b[i] - Ax[i];
        }
        return r;
    }

    static bool is_finite(const std::vector<double>& vec) {
        for (double v : vec) {
            if (!std::isfinite(v)) return false;
        }
        return true;
    }

    static bool is_zero(double val, double tol = 1e-9) {
        return std::abs(val) <= tol;
    }
};

} // namespace bharatopt
