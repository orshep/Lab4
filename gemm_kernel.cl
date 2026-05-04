// gemm_kernel.cl
__kernel void gemm_kernel(__global const float *input,
                     __global const float *weights,
                     const int input_size,
                     const int output_size,
                     __global float *output)
{
    int gid = get_global_id(0);
    if (gid < output_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[gid * input_size + i];
        }
        output[gid] = sum;
    }
}