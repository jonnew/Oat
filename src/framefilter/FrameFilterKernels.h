#ifndef OAT_FRAMEFILTERKERNELS_H
#define	OAT_FRAMEFILTERKERNELS_H

#include "../../lib/cv-cuda/CUDAHelpers.h"

//namespace oat {

__global__ 
void frameSubtract(cv::cuda::PtrStepSz<uchar3> src0,
                   cv::cuda::PtrStep<uchar3> src1,
                   cv::cuda::PtrStep<uchar3> dst);


__global__
void undistort(cv::cuda::PtrStepSz<uchar3> src,
               cv::cuda::PtrStep<uchar3> dst);

//}      /* namespace oat */
#endif /* OAT_FRAMEFILTERKERNELS_H */
