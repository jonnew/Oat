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

#ifdef OAT_USE_CUDA
#include <opencv2/cudabgsegm.hpp>
#else
#include <opencv2/video.hpp>
#endif

#include "FrameFilter.h"

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

    void configure(const std::string& config_file, const std::string& config_key);

private:

    /**
     * Apply background subtraction.
     * @param frame unfiltered frame
     * @return filtered frame
     */
    void filter(cv::Mat& frame);

#ifdef OAT_USE_CUDA
    cv::Ptr<cv::cuda::BackgroundSubtractorMOG> background_subtractor;
    cv::cuda::GpuMat current_frame, background_mask;
#else
    cv::Ptr<cv::BackgroundSubtractorMOG2> background_subtractor;
    cv::Mat background_mask;
#endif

    double learning_coeff {0.0};
};

#endif	/* OAT_BACKGROUNDSUBTRACTORMOG_H */

