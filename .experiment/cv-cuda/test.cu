#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda.hpp>
#include "CUDAHelpers.h"
#include <math.h>
#include <vector>

__global__ 
void testker(const cv::cuda::PtrStepSzb input,
        cv::cuda::PtrStepSzb output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= input.cols - 1 && 
            y <= input.rows - 1 && 
            y >= 0              && 
            x >= 0)
    {
        output(y, x) = input(y, x) / 2;
    }
}


int main()
{
    cv::Mat input = cv::imread("ada.jpg", 0);

    cv::cuda::GpuMat d_frame, d_output;
    d_frame.upload(input);

    callKernel(testker, d_frame, d_output, cv::cuda::Stream());

    cv::Mat output(d_output);


    std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);
    cv::imwrite("output.png", output, compression_params);
    return 0;
}
