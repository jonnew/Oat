//******************************************************************************
//* File:   BackgroundSubtractorMOG.cpp
//* Author: Jon Newman <jpnewman snail mit dot edu>
//*
//* Copyright (c) Jon Newman (jpnewman snail mit dot edu) 
//* All right reserved.
//* This file is part of the Simple Tracker project.
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

#ifndef BACKGROUNDSUBTRACTORMOG_H
#define	BACKGROUNDSUBTRACTORMOG_H

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
     * @param source_name raw frame source name
     * @param sink_name filtered frame sink name
     */
    BackgroundSubtractorMOG(const std::string& source_name, const std::string& sink_name);

    void configure(const std::string& config_file, const std::string& config_key);
    
private:
    
    /**
     * Apply background subtraction.
     * @param frame unfiltered frame
     * @return filtered frame
     */
    cv::Mat filter(cv::Mat& frame);

#ifdef OAT_USE_CUDA
    cv::Ptr<cv::cuda::BackgroundSubtractorMOG> background_subtractor;
    cv::cuda::GpuMat current_frame, background_mask;
#else
    cv::Ptr<cv::BackgroundSubtractorMOG2> background_subtractor;
    cv::Mat background_mask;
#endif
    
    double learning_coeff;

};

#endif	/* BACKGROUNDSUBTRACTORMOG_H */

