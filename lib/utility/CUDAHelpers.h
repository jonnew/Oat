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

// TODO: to implementation
#include <cuda_runtime.h>

#ifndef OAT_CUDAHELPERS_H
#define	OAT_CUDAHELPERS_H

#define MATIO_CU_KERNEL void(kernel*)   \
        (                               \
            const cv::cuda::PtrStepSzb, \
            cv::cuda::PtrStepSzb        \
        )

namespace oat {

int findCudaCapableDevice(int argc, char **argv);

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

    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(stream);
    (*kernel)<<<cblocks, cthreads, 0, stream>>>(input, output);

    cudaSafeCall(cudaGetLastError());
}

}      /* namespace oat */
#endif /* OAT_FILEFORMAT_H */
