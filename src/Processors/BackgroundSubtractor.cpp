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

#include <iostream>

using cv::Mat;

/**
 * Set the background image to be used during subsequent subtraction operations.
 * 
 * @param input_img The image to use as background.
 */
void BackgroundSubtractor::setBackgroundImage(const Mat& input_img) {

    background_set = true;
    background_img = input_img.clone();
}

/**
 * Subtract a previously set background image from an input image to produce
 * the output matrix.
 * 
 * @param input_img Input matrix
 * @param output_img Output matrix
 * 
 */
void BackgroundSubtractor::subtractBackground(const Mat& input_img, Mat& output_img) {

    // If we have set a background image, perform subtraction
    if (background_set) {

        if ((input_img.size() != output_img.size()) || (input_img.size() != background_img.size())) {
            std::cerr << "Background Subtractor: Input and output matrices must be the same size." << std::endl;
            exit(EXIT_FAILURE);
        }
        
        //cv::Mat temp = input_img.clone();
        output_img = input_img - background_img;
    } else {
        input_img.copyTo(output_img);
    }
    
    if (show) {
        showImage("Background subtraction", output_img);
    }
}
