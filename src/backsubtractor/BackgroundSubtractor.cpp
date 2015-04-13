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
#include "../../lib/shmem/MatClient.cpp"
#include "../../lib/shmem/MatServer.h"
#include "../../lib/shmem/MatServer.cpp"


using namespace boost::interprocess;

BackgroundSubtractor::BackgroundSubtractor(const std::string source_name, const std::string sink_name) :
    frame_source(source_name),
    frame_sink(sink_name) { }

/**
 * Set the background image to be used during subsequent subtraction operations.
 * The frame_source must have previously populated the the shared cv::Mat object.
 * 
 */
void BackgroundSubtractor::setBackgroundImage() {
    
    background_img = frame_source.get_shared_mat().clone();
    background_set = true;
    //subtractBackground(); // Need to call this for the wait(), which releases our
                          // shared lock on frame_source's mutex
}

/**
 * Subtract a previously set background image from an input image to produce
 * the output matrix.
 * 
 */
void BackgroundSubtractor::subtractBackground() {

    // If we have set a background image, perform subtraction
    cv::Mat current_frame = frame_source.get_shared_mat().clone();
    
    if (background_set) {

        if ((current_frame.size() != background_img.size())) {
            std::cerr << "Background subtractor: Input and background matrices must be the same size." << std::endl;
            exit(EXIT_FAILURE);
        }
        
        current_frame = current_frame - background_img;
    } 
    else {
     
        // First image is always used as the default background image
        setBackgroundImage(); 
    }
    
    frame_sink.set_shared_mat(current_frame);
    frame_source.wait();

}
