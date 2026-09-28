#pragma once
#include <vector>

void cusparse_spmv(int m, int n, int nnz, 
                   const int* csrRowPtrA, const int* csrColIndA, const double* csrValA, 
                   const double* x, double* y);

// Persistent PDLP GPU Context
struct GpuPdlpContext {
    void* handle; 
    void* matA;   
    void* vecX;   
    void* vecY;
    void* vecX_next;
    void* vecAx_next;
    void* vecAx;
    void* vecATy;
    
    int* d_csrRowPtrA;
    int* d_csrColIndA;
    double* d_csrValA;
    
    double* d_x;
    double* d_y;
    double* d_Ax;
    double* d_ATy;
    double* d_full_c;
    double* d_b;
    double* d_x_next;
    double* d_Ax_next;
    
    void* dBufferAx;
    void* dBufferATy;
    int m, n;
};

GpuPdlpContext* init_gpu_pdlp(int m, int n, int nnz,
                              const int* csrRowPtrA, const int* csrColIndA, const double* csrValA);

// Fully fused and asynchronous GPU execution loop
int run_pdlp_loop_gpu(GpuPdlpContext* ctx, 
                      const double* host_full_c, const double* host_b, 
                      double* host_x, double* host_y, double* host_Ax,
                      double tau, double sigma, int max_iters, int check_freq,
                      double& out_gap, double& out_rp, double& out_rd);

void free_gpu_pdlp(GpuPdlpContext* ctx);
