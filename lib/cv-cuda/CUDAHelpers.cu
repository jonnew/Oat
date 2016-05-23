//******************************************************************************
//* File:   CUDAHelpers.cu
//* Author: Jon Newman <jpnewman snail mit dot edu>
//*
//* Copyright (c) Jon Newman (jpnewman snail mit dot edu)
//* All right reserved.
//* This file is part of the Oat project.
//* This is free software: you can redistribute it and/or modify
//* it under the terms of the GNU General Public License as published by
//* the Free Software Foundation, either version 3 of the License, or
//* (at your option) any later version.
//* This software is distributed in the hope that it will be useful,
//* but WITHOUT ANY WARRANTY; without even the implied warranty of
//* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//* GNU General Public License for more details.
//* You should have received a copy of the GNU General Public License
//* along with this source code.  If not, see <http://www.gnu.org/licenses/>.
//*******************************************************************************

#include "CUDAHelpers.h"

#include <opencv2/core.hpp>
//#include <opencv2/core/cuda.hpp>
//#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

namespace oat {

template<typename T>
void callKernel(
    KernT0<T> kernel,
    const cv::cuda::GpuMat &src,
    cv::cuda::GpuMat &dst,
    cv::cuda::Stream &stream)
{

    dim3 block(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 grid((src.cols + block.x - 1)/block.x, (src.rows + block.y - 1)/ block.y);

    // kernel: FP to cuda kernel
    cudaStream_t s = cv::cuda::StreamAccessor::getStream(stream);
    (kernel)<<<grid, block, 0, s>>>(src, dst);

    if (s == 0)
        cudaDeviceSynchronize();
}

template<typename T>
void callKernel(
    KernT1<T> kernel,
    const cv::cuda::GpuMat &src0,
    const cv::cuda::GpuMat &src1,
    cv::cuda::GpuMat &dst,
    cv::cuda::Stream &stream)
{

    dim3 block(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 grid((src0.cols + block.x - 1)/block.x, (src0.rows + block.y - 1)/ block.y);

    // kernel: FP to cuda kernel
    cudaStream_t s = cv::cuda::StreamAccessor::getStream(stream);
    (kernel)<<<grid, block, 0, s>>>(src0, src1, dst);

    if (s == 0)
        cudaDeviceSynchronize();
}

// Explicit instantiations

template void callKernel<uchar3>(
    KernT0<uchar3> kernel,
    const cv::cuda::GpuMat &src,
    cv::cuda::GpuMat &dst,
    cv::cuda::Stream &stream
);

template void callKernel<uchar3>(
    KernT1<uchar3> kernel,
    const cv::cuda::GpuMat &src0,
    const cv::cuda::GpuMat &src1,
    cv::cuda::GpuMat &dst,
    cv::cuda::Stream &stream
);

} /* namespace oat */

