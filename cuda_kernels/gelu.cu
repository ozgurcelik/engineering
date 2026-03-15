#include <math.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

// __global__ means this function will be executed on the gpu
__global__ void gelu_kernel(float *x, float *y, int num_elements) {
    // Get the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Check if the thread is within the range of the input
    if (idx < num_elements) {
        // Apply the GELU activation function
        y[idx] = 0.5 * x[idx] * (1 + tanh(0.79788456 * (x[idx] + 0.044715 * x[idx] * x[idx] * x[idx])));
    }
}


inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

torch::Tensor gelu(torch::Tensor x) {
    // This is a wrapper that lives in the cpu and will orchestrate the kernel launch on the gpu
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    // Allocate empty output tensor dimension of x
    torch::Tensor y = torch::empty_like(x);

    // Determine the grid and block sizes
    int num_elements = x.numel();
    int block_size = 1024; // Number of threads per block
    int num_blocks = cdiv(num_elements, block_size);

    // Launch the kernel
    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), num_elements);
    C10_CUDA_KERNEL_LAUNCH_CHECK();  // Catch errors immediately

    // Return the output tensor
    return y;
}

