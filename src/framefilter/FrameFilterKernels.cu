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

typedef struct {
    float3 rad  {0.0};
    float2 tang {0.0};
} DistortionCoefficients;

typedef struct { 
    float2 focal_len {0.0};
    float2 inv_focal_len {0.0};
    float2 center    {0.0};
} CameraMatrix;

__constant__ const DistortionCoefficients * dc;
__constant__ const CameraMatrix * camera;

// TODO: Radial undistortion only
__global__
void undistort(cv::cuda::PtrStepSz<uchar3> src,
               cv::cuda::PtrStep<uchar3> dst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < src.cols && y < src.rows) {

        float2 idx = make_float2((float)x, (float)y);
        float2 lens_coords =
            idx.x - camera->center.x / camera - 
            (xy - camera->center) / camera->focal_len;

        float r2 = lens_coords.x * lens_coords.x + 
                   lens_coords.y * lens_coords.y;
        float r4 = r2 * r2; 
        float r6 = r4 * r2;

        float2 lens_undist = lens_coords * 
            (1 + dc->rad.x * r2 + dc->rad.y * r4 + dc->rad.z * r6);

        int2 uv_undist = 
            (camera->focal_len * lens_undist + optical_center);
       
        if (uv_undist.x < dst.cols && uv_undist.x >=0 && 
            uv_undist.y < dst.rows && uv_undist.y >=0) {
            dst(uv_undist.x, uv_undist.y) = src(x, y);
        }
    }
}

//} /* namespace oat */
