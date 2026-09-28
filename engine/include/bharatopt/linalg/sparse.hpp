#pragma once
#include <vector>
#include <cassert>
#include <stdexcept>
#include <cmath>

namespace bharatopt {

struct SparseStats {
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    double density = 0.0;
};

class CSRMatrix;
class CSCMatrix;

class COOMatrix {
public:
    COOMatrix(int rows, int cols) : rows_(rows), cols_(cols) {}

    void add_value(int row, int col, double val) {
        if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
            throw std::out_of_range("COO index out of bounds");
        }
        if (std::abs(val) > 1e-14) {
            row_indices_.push_back(row);
            col_indices_.push_back(col);
            values_.push_back(val);
        }
    }

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int nnz() const { return values_.size(); }

    const std::vector<int>& row_indices() const { return row_indices_; }
    const std::vector<int>& col_indices() const { return col_indices_; }
    const std::vector<double>& values() const { return values_; }

    SparseStats stats() const {
        return {rows_, cols_, nnz(), static_cast<double>(nnz()) / (rows_ * cols_)};
    }

private:
    int rows_;
    int cols_;
    std::vector<int> row_indices_;
    std::vector<int> col_indices_;
    std::vector<double> values_;
};

class CSRMatrix {
public:
    CSRMatrix(const COOMatrix& coo) : rows_(coo.rows()), cols_(coo.cols()) {
        row_ptr_.assign(rows_ + 1, 0);
        for (int r : coo.row_indices()) {
            row_ptr_[r + 1]++;
        }
        for (int i = 0; i < rows_; ++i) {
            row_ptr_[i + 1] += row_ptr_[i];
        }
        
        col_indices_.resize(coo.nnz());
        values_.resize(coo.nnz());
        
        std::vector<int> current_row_ptr = row_ptr_;
        for (size_t i = 0; i < coo.nnz(); ++i) {
            int r = coo.row_indices()[i];
            int dest = current_row_ptr[r]++;
            col_indices_[dest] = coo.col_indices()[i];
            values_[dest] = coo.values()[i];
        }
    }

    std::vector<double> spmv(const std::vector<double>& x) const {
        assert(x.size() == cols_);
        std::vector<double> y(rows_, 0.0);
        for (int i = 0; i < rows_; ++i) {
            for (int j = row_ptr_[i]; j < row_ptr_[i + 1]; ++j) {
                y[i] += values_[j] * x[col_indices_[j]];
            }
        }
        return y;
    }

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int nnz() const { return values_.size(); }

private:
    int rows_;
    int cols_;
    std::vector<int> row_ptr_;
    std::vector<int> col_indices_;
    std::vector<double> values_;
};

class CSCMatrix {
public:
    CSCMatrix(const COOMatrix& coo) : rows_(coo.rows()), cols_(coo.cols()) {
        col_ptr_.assign(cols_ + 1, 0);
        for (int c : coo.col_indices()) {
            col_ptr_[c + 1]++;
        }
        for (int i = 0; i < cols_; ++i) {
            col_ptr_[i + 1] += col_ptr_[i];
        }
        
        row_indices_.resize(coo.nnz());
        values_.resize(coo.nnz());
        
        std::vector<int> current_col_ptr = col_ptr_;
        for (size_t i = 0; i < coo.nnz(); ++i) {
            int c = coo.col_indices()[i];
            int dest = current_col_ptr[c]++;
            row_indices_[dest] = coo.row_indices()[i];
            values_[dest] = coo.values()[i];
        }
    }

    CSCMatrix transpose() const {
        // Simple transpose creates a CSR logically, which maps back to COO then CSC easily
        COOMatrix t_coo(cols_, rows_);
        for (int c = 0; c < cols_; ++c) {
            for (int j = col_ptr_[c]; j < col_ptr_[c + 1]; ++j) {
                t_coo.add_value(c, row_indices_[j], values_[j]);
            }
        }
        return CSCMatrix(t_coo);
    }
    
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int nnz() const { return values_.size(); }

private:
    int rows_;
    int cols_;
    std::vector<int> col_ptr_;
    std::vector<int> row_indices_;
    std::vector<double> values_;
};

} // namespace bharatopt
