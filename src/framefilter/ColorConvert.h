//******************************************************************************
//* File:   ColorConvert.h
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

#ifndef OAT_COLORCONVERT_H
#define	OAT_COLORCONVERT_H

#include "FrameFilter.h"

#include <opencv2/imgproc.hpp>

namespace oat {

/**
 * A frame color converter
 */
class ColorConvert : public FrameFilter {
public:

    /**
     * A basic background subtractor.
     * Subtract a frame image from a frame stream. The background frame is
     * the first frame obtained from the SOURCE frame stream, or can be
     * supplied via configuration file.
     * @param frame_source_address raw frame source address
     * @param frame_sink_address filtered frame sink address
     */
    ColorConvert(const std::string &frame_souce_address,
                 const std::string &frame_sink_address);

    void connectToNode(void) override;
    void configure(const std::string &config_file,
                   const std::string &config_key) override;

private:

    /**
     * Apply color conversion 
     * @param frame unfiltered frame
     * @return filtered frame
     */
    void filter(cv::Mat& frame) override;

    /// OpenCV color conversion code hash
    //static const std::unordered_map<std::string, cv::ColorConversionCodes> 
    //color_conversion_map_;
    
    cv::ColorConversionCodes conversion_code_ = cv::COLOR_BGR2GRAY; 
};

}      /* namespace oat */
#endif /* OAT_COLORCONVERT_H */


