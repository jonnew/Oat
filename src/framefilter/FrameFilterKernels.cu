#include "FrameFilterKernels.h"

//namespace oat {

__global__
void frameSubtract(cv::cuda::PtrStepSz<uchar3> src0,
                   cv::cuda::PtrStep<uchar3> src1,
                   cv::cuda::PtrStep<uchar3> dst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < src0.cols && y < src0.rows) {
        uchar3 v0 = src0(x, y);
        uchar3 v1 = src1(x, y);
        dst(x, y) = make_uchar3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
    }
}

//typedef struct {
//    int     n_radial;
//    int     n_tangential;
//    float * radial[5];
//    float * tangential[2];
//} DistortionCoefficients;
//
//typedef struct { 
//    int width; 
//    int height; 
//    float * elements; 
//} Matrix;
//
//__constant__ const DistortionCoefficients dc;
//__constant__ const Matrix camera_mat;

__global__
void undistort(cv::cuda::PtrStepSz<uchar3> src,
               cv::cuda::PtrStep<uchar3> dst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < src.cols && y < src.rows) {
        uchar3 v = src(x, y);
        dst(x, y) = v;
    }
}

//} /* namespace oat */
