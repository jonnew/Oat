//******************************************************************************
//* File:   FrameMasker.h
//* Author: Jon Newman <jpnewman snail mit dot edu>
//
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

#ifndef FRAMEMASKER_H
#define	FRAMEMASKER_H

#include "FrameFilter.h"

/**
 * A frame masker.
 */
class FrameMasker : public FrameFilter {
public:
    
    /**
     * A frame masker.
     * A frame masker to isolate one or more regions of interest in a frame stream using
     * a mask frame. Pixels of the input frames that correspond to non-zero pixels in
     * the mask frame will be unchanged. All other pixels will be set to 0. 
     * @param source_name raw frame source name
     * @param sink_name filtered frame sink name
     * @param invert_mask invert the mask frame before filtering.
     */
    FrameMasker(const std::string& source_name, 
                const std::string& sink_name, 
                bool invert_mask=false);
    
    void configure(const std::string& config_file, const std::string& config_key);
    
private:
    
    /**
     * Apply frame mask.
     * @param frame unfiltered frame
     * @return filtered frame
     */
    cv::Mat filter(cv::Mat& frame);

    // Should be inverted before application.
    bool invert_mask;
    
    // Do we have a mask to work with
    bool mask_set = false;
    
    // Mask frames with an arbitrary ROI
    cv::Mat roi_mask;
};

#endif	/* FRAMEMASKER_H */

