#include "CUDAHelpers.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

void callKernel(MATIO_CU_KERNEL,
                cv::InputArray _input,
                cv::OutputArray _output,
                cv::cuda::Stream _stream)
{

    // Automatic conversion to GPU mat
    // TODO: Efficient?
    const cv::cuda::GpuMat input = _input.getGpuMat();

    _output.create(input.size(), input.type());
    cv::cuda::GpuMat output = _output.getGpuMat();

    // TODO: Optimize?
    dim3 cthreads(16, 16);
    dim3 cblocks(
        static_cast<int>(std::ceil(input.size().width /
            static_cast<double>(cthreads.x))),
        static_cast<int>(std::ceil(input.size().height /
            static_cast<double>(cthreads.y))));

    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(_stream);
    (kernel)<<<cblocks, cthreads, 0, stream>>>(input, output);

    cudaSafeCall(cudaGetLastError());
}
