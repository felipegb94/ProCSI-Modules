#include <iostream>
#include <cmath>
#include <string>

#include <thrust/reduce.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>

const int ARRAY_SIZE = 1000;

int main(int argc, char **argv) {
    double* mA;
    cudaMallocManaged(&mA, ARRAY_SIZE * sizeof(double));
    thrust::sequence(mA, mA + ARRAY_SIZE, 1);

    /* thrust::reduce(thrust::cuda/omp/cpp::par,start, end, operations) */
    double maximumGPU = thrust::reduce(thrust::cuda::par, mA, mA + ARRAY_SIZE, 0.0,      
                                       thrust::maximum<double>());
    cudaDeviceSynchronize();
    double maximumCPU = thrust::reduce(thrust::omp::par, mA, mA + ARRAY_SIZE, 0.0,    
                                       thrust::maximum<double>());

    std::string gpuPassed = "Failed";
    std::string cpuPassed = "Failed";

    if(std::fabs(maximumGPU - ARRAY_SIZE) < (1e-10))
    {
        gpuPassed = "Passed";
    }
    if(std::fabs(maximumCPU - ARRAY_SIZE) < (1e-10))
    {
        cpuPassed = "Passed";
    }

    std::cout << "GPU reduce: " << gpuPassed << std::endl;
    std::cout << "CPU reduce: " << cpuPassed << std::endl;
    cudaFree(mA);

    return 0;
}