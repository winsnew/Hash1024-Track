#include "protobuff.h"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <fstream>

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx] + 1.0f; 
    }
}


__global__ void matrixVectorMult(const float* matrix, const float* vector, 
                                float* result, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        float sum = 0.0f;
        for (int col = 0; col < cols; col++) {
            sum += matrix[row * cols + col] * vector[col];
        }
        result[row] = sum;
    }
}

__global__ void reduceSum(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

CudaProtobufHandler::CudaProtobufHandler() {
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?" << std::endl;
    }
}

CudaProtobufHandler::~CudaProtobufHandler() {
    cudaDeviceReset();
}

bool CudaProtobufHandler::serializeToFile(const cuda_example::VectorData& data, 
                                         const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return false;
    }
    
    if (!data.SerializeToOstream(&file)) {
        std::cerr << "Failed to serialize data to file." << std::endl;
        return false;
    }
    
    file.close();
    return true;
}

bool CudaProtobufHandler::deserializeFromFile(const std::string& filename, 
                                             cuda_example::VectorData& data) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return false;
    }
    
    if (!data.ParseFromIstream(&file)) {
        std::cerr << "Failed to parse data from file." << std::endl;
        return false;
    }
    
    file.close();
    return true;
}

float* CudaProtobufHandler::copyToDevice(const float* host_data, size_t size) {
    float* device_ptr = nullptr;
    cudaError_t err = cudaMalloc(&device_ptr, size * sizeof(float));
    
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }
    
    err = cudaMemcpy(device_ptr, host_data, size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy host to device failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(device_ptr);
        return nullptr;
    }
    
    return device_ptr;
}

void CudaProtobufHandler::copyToHost(float* host_dest, const float* device_src, size_t size) {
    cudaError_t err = cudaMemcpy(host_dest, device_src, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy device to host failed: " << cudaGetErrorString(err) << std::endl;
    }
}

void CudaProtobufHandler::freeDeviceMemory(float* device_ptr) {
    if (device_ptr) {
        cudaFree(device_ptr);
    }
}

// Processing
cuda_example::ComputationResult CudaProtobufHandler::processVectorOnGPU(const cuda_example::VectorData& input) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    cuda_example::ComputationResult result;
    int size = input.size();
    
    float* d_input = copyToDevice(input.values().data(), size);
    float* d_output = nullptr;
    cudaMalloc(&d_output, size * sizeof(float));
    
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    vectorAdd<<<numBlocks, blockSize>>>(d_input, d_input, d_output, size);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Vector add kernel failed: " << cudaGetErrorString(err) << std::endl;
    }
    
    cudaDeviceSynchronize();
    
    std::vector<float> host_result(size);
    copyToHost(host_result.data(), d_output, size);
    
    auto* processed_vector = result.mutable_processed_vector();
    processed_vector->set_size(size);
    for (int i = 0; i < size; i++) {
        processed_vector->add_values(host_result[i]);
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += host_result[i];
    }
    result.set_result(sum);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    result.set_computation_time(duration.count() / 1000.0); // Convert to milliseconds

    freeDeviceMemory(d_input);
    freeDeviceMemory(d_output);
    
    return result;
}

// Matrix multiplication
cuda_example::ComputationResult CudaProtobufHandler::matrixVectorMultiply(
    const cuda_example::MatrixData& matrix, 
    const cuda_example::VectorData& vector) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    cuda_example::ComputationResult result;
    int rows = matrix.rows();
    int cols = matrix.cols();
    
    if (vector.size() != cols) {
        std::cerr << "Dimension mismatch: matrix cols (" << cols 
                  << ") != vector size (" << vector.size() << ")" << std::endl;
        return result;
    }
    
    float* d_matrix = copyToDevice(matrix.elements().data(), rows * cols);
    float* d_vector = copyToDevice(vector.values().data(), cols);
    float* d_result = nullptr;
    cudaMalloc(&d_result, rows * sizeof(float));
    
    int blockSize = 256;
    int numBlocks = (rows + blockSize - 1) / blockSize;
    
    // Launch kernel
    matrixVectorMult<<<numBlocks, blockSize>>>(d_matrix, d_vector, d_result, rows, cols);
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Matrix-vector kernel failed: " << cudaGetErrorString(err) << std::endl;
    }
    
    cudaDeviceSynchronize();
    
    // Copy result back to host
    std::vector<float> host_result(rows);
    copyToHost(host_result.data(), d_result, rows);
    
    // Create result proto
    auto* processed_vector = result.mutable_processed_vector();
    processed_vector->set_size(rows);
    for (int i = 0; i < rows; i++) {
        processed_vector->add_values(host_result[i]);
    }
    
    // Calculate norm as example result
    float norm = 0.0f;
    for (int i = 0; i < rows; i++) {
        norm += host_result[i] * host_result[i];
    }
    result.set_result(sqrt(norm));

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    result.set_computation_time(duration.count() / 1000.0);
    freeDeviceMemory(d_matrix);
    freeDeviceMemory(d_vector);
    freeDeviceMemory(d_result);
    
    return result;
}