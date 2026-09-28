#pragma once
#include <vector>
#include <iostream>

namespace bharatopt {

struct BackendResult {
    double gpu_kernel_time = 0.0;
    double h2d_time = 0.0;
    double d2h_time = 0.0;
    double total_gpu_time = 0.0;
    double cpu_time = 0.0;
    double speedup = 1.0;
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual bool is_available() const = 0;
    virtual std::vector<double> spmv(const std::vector<double>& values,
                                     const std::vector<int>& col_indices,
                                     const std::vector<int>& row_ptr,
                                     const std::vector<double>& x) = 0;
    virtual BackendResult last_result() const = 0;
};

class CPUBackend : public Backend {
public:
    bool is_available() const override { return true; }

    std::vector<double> spmv(const std::vector<double>& values,
                             const std::vector<int>& col_indices,
                             const std::vector<int>& row_ptr,
                             const std::vector<double>& x) override {
        // Basic CSR SpMV on CPU
        int rows = row_ptr.size() - 1;
        std::vector<double> y(rows, 0.0);
        for (int i = 0; i < rows; ++i) {
            for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
                y[i] += values[j] * x[col_indices[j]];
            }
        }
        last_res_.cpu_time = 0.01; // Mock timing
        return y;
    }

    BackendResult last_result() const override { return last_res_; }

private:
    BackendResult last_res_;
};

class CUDABackend : public Backend {
public:
    bool is_available() const override {
#ifdef USE_CUDA
        return true;
#else
        return false;
#endif
    }

    std::vector<double> spmv(const std::vector<double>& values,
                             const std::vector<int>& col_indices,
                             const std::vector<int>& row_ptr,
                             const std::vector<double>& x) override {
        if (!is_available()) {
            throw std::runtime_error("CUDA not available. Compile with USE_CUDA.");
        }
        
        // Mock CUDA SpMV returning correct data structures
        int rows = row_ptr.size() - 1;
        std::vector<double> y(rows, 0.0);
        
        last_res_.h2d_time = 0.001;
        last_res_.gpu_kernel_time = 0.005;
        last_res_.d2h_time = 0.001;
        last_res_.total_gpu_time = 0.007;
        last_res_.cpu_time = 0.01;
        last_res_.speedup = last_res_.cpu_time / last_res_.total_gpu_time;
        
        return y;
    }

    BackendResult last_result() const override { return last_res_; }

private:
    BackendResult last_res_;
};

} // namespace bharatopt
