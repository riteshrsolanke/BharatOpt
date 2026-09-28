#include "gpu_kernels.h"
#include <iostream>

#if defined(USE_CUDA) && USE_CUDA == 1
#include <cuda_runtime.h>
#include <cusparse.h>

void cusparse_spmv(int m, int n, int nnz, 
                   const int* csrRowPtrA, const int* csrColIndA, const double* csrValA, 
                   const double* x, double* y) {
    // Basic cuSPARSE SpMV setup
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    
    // Allocate device memory
    int* d_csrRowPtrA;
    int* d_csrColIndA;
    double* d_csrValA;
    double* d_x;
    double* d_y;
    
    cudaMalloc((void**)&d_csrRowPtrA, (m + 1) * sizeof(int));
    cudaMalloc((void**)&d_csrColIndA, nnz * sizeof(int));
    cudaMalloc((void**)&d_csrValA, nnz * sizeof(double));
    cudaMalloc((void**)&d_x, n * sizeof(double));
    cudaMalloc((void**)&d_y, m * sizeof(double));
    
    cudaMemcpy(d_csrRowPtrA, csrRowPtrA, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIndA, csrColIndA, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrValA, csrValA, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);
    
    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA, m, n, nnz, d_csrRowPtrA, d_csrColIndA, d_csrValA, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    
    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, n, d_x, CUDA_R_64F);
    cusparseCreateDnVec(&vecY, m, d_y, CUDA_R_64F);
    
    double alpha = 1.0;
    double beta = 0.0;
    size_t bufferSize = 0;
    void* dBuffer = NULL;
    
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
    
    cudaMemcpy(y, d_y, m * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(dBuffer);
    cudaFree(d_csrRowPtrA); cudaFree(d_csrColIndA); cudaFree(d_csrValA);
    cudaFree(d_x); cudaFree(d_y);
    cusparseDestroySpMat(matA); cusparseDestroyDnVec(vecX); cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);
}

__global__ void pdlp_update_x_kernel(int n, const double* x, const double* full_c, const double* ATy, double tau, double* x_next) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double v = full_c[i] - ATy[i];
        double val = x[i] - tau * v;
        x_next[i] = val > 0.0 ? val : 0.0;
    }
}

__global__ void pdlp_update_y_kernel(int m, double* y, const double* Ax_next, double* Ax, const double* b, double sigma) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        double ax_n = Ax_next[i];
        y[i] = y[i] + sigma * (b[i] - 2.0 * ax_n + Ax[i]);
        Ax[i] = ax_n; // Update Ax for the next iteration
    }
}

GpuPdlpContext* init_gpu_pdlp(int m, int n, int nnz,
                              const int* csrRowPtrA, const int* csrColIndA, const double* csrValA) {
    GpuPdlpContext* ctx = new GpuPdlpContext();
    ctx->m = m;
    ctx->n = n;
    cusparseCreate((cusparseHandle_t*)&ctx->handle);
    
    cudaMalloc((void**)&ctx->d_csrRowPtrA, (m + 1) * sizeof(int));
    cudaMalloc((void**)&ctx->d_csrColIndA, nnz * sizeof(int));
    cudaMalloc((void**)&ctx->d_csrValA, nnz * sizeof(double));
    cudaMalloc((void**)&ctx->d_x, n * sizeof(double));
    cudaMalloc((void**)&ctx->d_y, m * sizeof(double));
    cudaMalloc((void**)&ctx->d_Ax, m * sizeof(double));
    cudaMalloc((void**)&ctx->d_ATy, n * sizeof(double));
    cudaMalloc((void**)&ctx->d_full_c, n * sizeof(double));
    cudaMalloc((void**)&ctx->d_b, m * sizeof(double));
    cudaMalloc((void**)&ctx->d_x_next, n * sizeof(double));
    cudaMalloc((void**)&ctx->d_Ax_next, m * sizeof(double));
    
    cudaMemcpy(ctx->d_csrRowPtrA, csrRowPtrA, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_csrColIndA, csrColIndA, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_csrValA, csrValA, nnz * sizeof(double), cudaMemcpyHostToDevice);
    
    cusparseCreateCsr((cusparseSpMatDescr_t*)&ctx->matA, m, n, nnz, ctx->d_csrRowPtrA, ctx->d_csrColIndA, ctx->d_csrValA, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    
    cusparseCreateDnVec((cusparseDnVecDescr_t*)&ctx->vecX, n, ctx->d_x, CUDA_R_64F);
    cusparseCreateDnVec((cusparseDnVecDescr_t*)&ctx->vecY, m, ctx->d_y, CUDA_R_64F);
    cusparseCreateDnVec((cusparseDnVecDescr_t*)&ctx->vecX_next, n, ctx->d_x_next, CUDA_R_64F);
    cusparseCreateDnVec((cusparseDnVecDescr_t*)&ctx->vecAx_next, m, ctx->d_Ax_next, CUDA_R_64F);
    cusparseCreateDnVec((cusparseDnVecDescr_t*)&ctx->vecAx, m, ctx->d_Ax, CUDA_R_64F);
    cusparseCreateDnVec((cusparseDnVecDescr_t*)&ctx->vecATy, n, ctx->d_ATy, CUDA_R_64F);
    
    double alpha = 1.0, beta = 0.0;
    size_t bufferSizeAx = 0, bufferSizeATy = 0;
    cusparseSpMV_bufferSize((cusparseHandle_t)ctx->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, (cusparseSpMatDescr_t)ctx->matA, (cusparseDnVecDescr_t)ctx->vecX_next, &beta, (cusparseDnVecDescr_t)ctx->vecAx_next, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSizeAx);
    cudaMalloc(&ctx->dBufferAx, bufferSizeAx);
    
    cusparseSpMV_bufferSize((cusparseHandle_t)ctx->handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, (cusparseSpMatDescr_t)ctx->matA, (cusparseDnVecDescr_t)ctx->vecY, &beta, (cusparseDnVecDescr_t)ctx->vecATy, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSizeATy);
    cudaMalloc(&ctx->dBufferATy, bufferSizeATy);
    
    return ctx;
}

int run_pdlp_loop_gpu(GpuPdlpContext* ctx, 
                      const double* host_full_c, const double* host_b, 
                      double* host_x, double* host_y, double* host_Ax,
                      double tau, double sigma, int max_iters, int check_freq,
                      double& out_gap, double& out_rp, double& out_rd) {
    
    cudaMemcpy(ctx->d_full_c, host_full_c, ctx->n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_b, host_b, ctx->m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_x, host_x, ctx->n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_y, host_y, ctx->m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_Ax, host_Ax, ctx->m * sizeof(double), cudaMemcpyHostToDevice);
    
    double b_norm = 0.0;
    for (int i = 0; i < ctx->m; ++i) b_norm += host_b[i] * host_b[i];
    b_norm = sqrt(b_norm);
    
    double c_norm = 0.0;
    for (int j = 0; j < ctx->n; ++j) c_norm += host_full_c[j] * host_full_c[j];
    c_norm = sqrt(c_norm);

    std::vector<double> temp_ATy(ctx->n, 0.0);

    double alpha = 1.0, beta = 0.0;
    int blocks_n = (ctx->n + 255) / 256;
    int blocks_m = (ctx->m + 255) / 256;
    
    int iters = 0;
    for (iters = 0; iters < max_iters; ++iters) {
        // 1. ATy = A^T * y
        cusparseSpMV((cusparseHandle_t)ctx->handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, (cusparseSpMatDescr_t)ctx->matA, (cusparseDnVecDescr_t)ctx->vecY, &beta, (cusparseDnVecDescr_t)ctx->vecATy, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, ctx->dBufferATy);
        
        // 2. x_next = proj(x - tau * (full_c - ATy))
        pdlp_update_x_kernel<<<blocks_n, 256>>>(ctx->n, ctx->d_x, ctx->d_full_c, ctx->d_ATy, tau, ctx->d_x_next);
        
        // 3. Ax_next = A * x_next
        cusparseSpMV((cusparseHandle_t)ctx->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, (cusparseSpMatDescr_t)ctx->matA, (cusparseDnVecDescr_t)ctx->vecX_next, &beta, (cusparseDnVecDescr_t)ctx->vecAx_next, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, ctx->dBufferAx);
        
        // 4. y = y + sigma * (2 * Ax_next - Ax - b) AND Ax = Ax_next
        pdlp_update_y_kernel<<<blocks_m, 256>>>(ctx->m, ctx->d_y, ctx->d_Ax_next, ctx->d_Ax, ctx->d_b, sigma);
        
        // 5. x = x_next
        cudaMemcpyAsync(ctx->d_x, ctx->d_x_next, ctx->n * sizeof(double), cudaMemcpyDeviceToDevice);
        
        if (iters % check_freq == 0 && iters > 0) {
            cudaDeviceSynchronize();
            cudaMemcpy(host_x, ctx->d_x, ctx->n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(host_y, ctx->d_y, ctx->m * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(host_Ax, ctx->d_Ax, ctx->m * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(temp_ATy.data(), ctx->d_ATy, ctx->n * sizeof(double), cudaMemcpyDeviceToHost);
            
            double rp_sq = 0.0, rd_sq = 0.0, cx = 0.0, by = 0.0;
            for (int i = 0; i < ctx->m; ++i) {
                double diff = host_Ax[i] - host_b[i];
                rp_sq += diff * diff;
                by += host_b[i] * host_y[i];
            }
            for (int j = 0; j < ctx->n; ++j) {
                double diff = temp_ATy[j] - host_full_c[j];
                rd_sq += diff * diff;
                cx += host_full_c[j] * host_x[j];
            }
            double rp = sqrt(rp_sq) / (1.0 + b_norm);
            double rd = sqrt(rd_sq) / (1.0 + c_norm);
            double gap = fabs(cx - by) / (1.0 + fabs(cx) + fabs(by));
            
            out_gap = gap;
            out_rp = rp;
            out_rd = rd;
            
            if (rp < 1e-4 && rd < 1e-4 && gap < 1e-4) {
                return iters;
            }
        }
    }
    cudaDeviceSynchronize();
    
    cudaMemcpy(host_x, ctx->d_x, ctx->n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_y, ctx->d_y, ctx->m * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_Ax, ctx->d_Ax, ctx->m * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(temp_ATy.data(), ctx->d_ATy, ctx->n * sizeof(double), cudaMemcpyDeviceToHost);

    double rp_sq = 0.0, rd_sq = 0.0, cx = 0.0, by = 0.0;
    for (int i = 0; i < ctx->m; ++i) {
        double diff = host_Ax[i] - host_b[i];
        rp_sq += diff * diff;
        by += host_b[i] * host_y[i];
    }
    for (int j = 0; j < ctx->n; ++j) {
        double diff = temp_ATy[j] - host_full_c[j];
        rd_sq += diff * diff;
        cx += host_full_c[j] * host_x[j];
    }
    out_rp = sqrt(rp_sq) / (1.0 + b_norm);
    out_rd = sqrt(rd_sq) / (1.0 + c_norm);
    out_gap = fabs(cx - by) / (1.0 + fabs(cx) + fabs(by));
    
    return iters;
}

void free_gpu_pdlp(GpuPdlpContext* ctx) {
    if (!ctx) return;
    cudaFree(ctx->dBufferAx); cudaFree(ctx->dBufferATy);
    cudaFree(ctx->d_csrRowPtrA); cudaFree(ctx->d_csrColIndA); cudaFree(ctx->d_csrValA);
    cudaFree(ctx->d_x); cudaFree(ctx->d_y); cudaFree(ctx->d_Ax); cudaFree(ctx->d_ATy);
    cudaFree(ctx->d_full_c); cudaFree(ctx->d_b); cudaFree(ctx->d_x_next); cudaFree(ctx->d_Ax_next);
    cusparseDestroySpMat((cusparseSpMatDescr_t)ctx->matA);
    cusparseDestroyDnVec((cusparseDnVecDescr_t)ctx->vecX); cusparseDestroyDnVec((cusparseDnVecDescr_t)ctx->vecY);
    cusparseDestroyDnVec((cusparseDnVecDescr_t)ctx->vecX_next); cusparseDestroyDnVec((cusparseDnVecDescr_t)ctx->vecAx_next);
    cusparseDestroyDnVec((cusparseDnVecDescr_t)ctx->vecAx); cusparseDestroyDnVec((cusparseDnVecDescr_t)ctx->vecATy);
    cusparseDestroy((cusparseHandle_t)ctx->handle);
    delete ctx;
}

#else

void cusparse_spmv(int m, int n, int nnz, 
                   const int* csrRowPtrA, const int* csrColIndA, const double* csrValA, 
                   const double* x, double* y) {
    std::cout << "CUDA disabled. Fallback used." << std::endl;
}

GpuPdlpContext* init_gpu_pdlp(int m, int n, int nnz,
                              const int* csrRowPtrA, const int* csrColIndA, const double* csrValA) {
    std::cout << "CUDA disabled. Returning null context." << std::endl;
    return nullptr;
}

int run_pdlp_loop_gpu(GpuPdlpContext* ctx, 
                      const double* host_full_c, const double* host_b, 
                      double* host_x, double* host_y, double* host_Ax,
                      double tau, double sigma, int max_iters, int check_freq,
                      double& out_gap, double& out_rp, double& out_rd) {
    return 0; // Stub
}

void free_gpu_pdlp(GpuPdlpContext* ctx) {
    // Stub
}
#endif
