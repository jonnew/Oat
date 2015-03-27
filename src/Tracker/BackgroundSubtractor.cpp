/* 
 * File:   BackgroundSubtractor.cpp
 * Author: Jon Newman <jpnewman snail mit dot edu>
 * 
 * Created on March 26, 2015, 6:08 PM
 */

#include "BackgroundSubtractor.h"

#include <iostream>

using cv::Mat;

BackgroundSubtractor::BackgroundSubtractor() {
}

BackgroundSubtractor::~BackgroundSubtractor() {
}

/**
 * Set the background image to be used during subsequent subtraction operations.
 * 
 * @param input_img The image to use as background.
 */
void BackgroundSubtractor::setBackgroundImage(const Mat& input_img) {

    background_img = input_img.clone();
}

/**
 * Subtract a previously set background image from an input image to produce
 * the output matrix.
 * 
 * @param input_img Input matrix
 * @param output_img Output matrix
 */
void BackgroundSubtractor::subtrackBackground(Mat& input_img, Mat& output_img) {



    // If we have set a background image, perform subtraction
    if (&background_img) {

        if ((input_img.size() != output_img.size()) || (input_img.size() != background_img.size())) {
            std::cerr << "Input and output matrices must be the same size." << std::endl;
            exit(EXIT_FAILURE);
        }
        output_img = input_img - background_img;
    }
    else {
        output_img = input_img;
    }
}
