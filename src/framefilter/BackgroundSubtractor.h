//******************************************************************************
//* File:   BackgroundSubtractor.h
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

#ifndef BACKGROUNDSUBTRACTOR_H
#define	BACKGROUNDSUBTRACTOR_H

#include "FrameFilter.h"

class BackgroundSubtractor : public FrameFilter {
public:

    /**
     * A background subtractor.
     * Subtract a frame image from a frame stream. The background frame is 
     * the first frame obtained from the SOURCE frame stream, or can be 
     * supplied via configuration file.
     * @param source_name raw frame source name
     * @param sink_name filtered frame sink name
     */
    BackgroundSubtractor(const std::string& source_name, const std::string& sink_name);

    /**
     * Apply background subtraction.
     * @param frame unfiltered frame
     * @return filtered frame
     */
    cv::Mat filter(cv::Mat& frame);
    
    void configure(const std::string& config_file, const std::string& config_key);
    
private:

    // Is the background frame set?
    bool background_set = false;

    // The background frame
    cv::Mat background_img;
    
    // Set the background frame
    void setBackgroundImage(const cv::Mat&);

};

#endif	/* BACKGROUNDSUBTRACTOR_H */

