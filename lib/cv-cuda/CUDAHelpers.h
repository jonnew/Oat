//******************************************************************************
//* File:   CUDAHelpers.h
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

#ifndef OAT_CUDAHELPERS_H
#define	OAT_CUDAHELPERS_H

#include <cuda_runtime.h>
//#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda/common.hpp>
//#include <opencv2/core/cuda_stream_accessor.hpp>

namespace oat {

// TODO: Optimize for each kernel executation?
static const int THREADS_PER_BLOCK {16};

template<typename T>
using KernT0 = void (*)(
    const cv::cuda::PtrStepSz<T>,
    cv::cuda::PtrStepSz<T>
);

template<typename T>
using KernT1 = void (*)(
    const cv::cuda::PtrStepSz<T>,
    const cv::cuda::PtrStep<T>,
    cv::cuda::PtrStep<T>
);

template<typename T>
void callKernel(
    KernT0<T> kernel,
    const cv::cuda::GpuMat &src,
    cv::cuda::GpuMat &dst,
    cv::cuda::Stream &stream = cv::cuda::Stream::Null()
);

template<typename T>
void callKernel(
    KernT1<T> kernel,
    const cv::cuda::GpuMat &src0,
    const cv::cuda::GpuMat &src1,
    cv::cuda::GpuMat &dst,
    cv::cuda::Stream &stream = cv::cuda::Stream::Null()
);

}      /* namespace oat */
#endif /* OAT_FILEFORMAT_H */
