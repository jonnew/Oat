//******************************************************************************
//* Copyright (c) Jon Newman (jpnewman at mit snail edu) 
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

#include "BackgroundSubtractor.h"

#include <string>
#include <iostream>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

#include "../../lib/shmem/MatClient.h"
#include "../../lib/shmem/MatServer.h"

BackgroundSubtractor::BackgroundSubtractor(const std::string source_name, const std::string sink_name) :
frame_source(source_name)
, frame_sink(sink_name) { }

/**
 * Set the background image to be used during subsequent subtraction operations.
 * The frame_source must have previously populated the the shared cv::Mat object.
 * 
 */
void BackgroundSubtractor::setBackgroundImage() {

    background_img = current_raw_frame.clone();
    background_set = true;
}

/**
 * Subtract a previously set background image from an input image to produce
 * the output matrix.
 * 
 */
void BackgroundSubtractor::subtractBackground() {
    
    // Only proceed with processing if we are getting a valid frame
    if (frame_source.getSharedMat(current_frame)) {

        current_raw_frame = current_frame.clone();
        
        if (background_set) {

            try {
                CV_Assert(current_frame.size() == background_img.size());
                current_frame = current_frame - background_img;
            } catch (cv::Exception& e) {
                std::cout << "CV Exception: " << e.what() << "\n";
                exit(EXIT_FAILURE);
            }

        } else {

            // First image is always used as the default background image
            setBackgroundImage();
        }

        // Push filtered frame forward, along with frame_source sample number
        frame_sink.pushMat(current_frame, frame_source.get_current_time_stamp());
    }

}
