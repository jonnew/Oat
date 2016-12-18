//******************************************************************************
//* File:   BackgroundSubtractorMOG.cpp
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
//******************************************************************************

#ifndef OAT_BACKGROUNDSUBTRACTORMOG_H
#define	OAT_BACKGROUNDSUBTRACTORMOG_H

#include <opencv2/cvconfig.h>

#ifdef HAVE_CUDA
 #include <opencv2/cudabgsegm.hpp>
#else
 #include <opencv2/video.hpp>
#endif

#include "FrameFilter.h"

namespace oat {

/**
 * A MOG background subtractor.
 */
class BackgroundSubtractorMOG : public FrameFilter {
public:

    /**
     * A MOG background subtractor.
     * @param frame_source_address raw frame source address
     * @param frame_sink_address filtered frame sink address
     */
    BackgroundSubtractorMOG(const std::string &frame_souce_address,
                            const std::string &frame_sink_address);

private:
    po::options_description options() const override;
    void applyConfiguration(const po::variables_map &vm,
                            const config::OptionTable &config_table) override;

    /**
     * Apply background subtraction.
     * @param frame unfiltered frame
     * @return filtered frame
     */
    void filter(cv::Mat &frame) override;

#ifdef HAVE_CUDA

     /**
     * Configure the GPU to perform background subtraction.
     * @param index_ Index of the GPU to use for processing
     */
    void configureGPU(size_t index_);

    cv::Ptr<cv::cuda::BackgroundSubtractorMOG> background_subtractor_;
    cv::cuda::GpuMat current_frame_, background_mask_;
#else
    cv::Ptr<cv::BackgroundSubtractorMOG2> background_subtractor_;
    cv::Mat background_mask_;
#endif

    double learning_coeff_ {0.0};
};

}      /* namespace oat */
#endif /* OAT_BACKGROUNDSUBTRACTORMOG_H */
