#ifndef PROTOBUF_H
#define PROTOBUF_H

#include "example.pb.h"
#include <vector>

class CudaProtobufHandler {
public:
    CudaProtobufHandler();
    ~CudaProtobufHandler();
    bool serializeToFile(const cuda_example::VectorData& data, const std::string& filename);
    bool deserializeFromFile(const std::string& filename, cuda_example::VectorData& data);
    cuda_example::ComputationResult processVectorOnGPU(const cuda_example::VectorData& input);
    cuda_example::ComputationResult matrixVectorMultiply(const cuda_example::MatrixData& matrix, 
                                                        const cuda_example::VectorData& vector);

private:
    float* copyToDevice(const float* host_data, size_t size);
    void copyToHost(float* host_dest, const float* device_src, size_t size);
    void freeDeviceMemory(float* device_ptr);
    void vectorAddKernel(float* a, float* b, float* c, int n);
    void matrixVectorKernel(float* matrix, float* vector, float* result, int rows, int cols);
    float reduceSumKernel(float* data, int size);
};

#endif