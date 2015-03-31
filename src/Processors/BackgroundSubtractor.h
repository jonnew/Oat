/* 
 * File:   BackgroundSubtractor.h
 * Author: Jon Newman <jpnewman snail mit dot edu>
 *
 * Created on March 26, 2015, 6:08 PM
 */

#ifndef BACKGROUNDSUBTRACTOR_H
#define	BACKGROUNDSUBTRACTOR_H

#include "ImageDisplay.h"
#include <opencv2/core/mat.hpp>

class Tracker;

class BackgroundSubtractor : public ImageDisplay {
    
    friend Tracker;
    
public:

    void setBackgroundImage(const cv::Mat& input_img);
    void subtractBackground(const cv::Mat& input_img, cv::Mat& output_img);
    
private:
    
    // The background image used for subtraction
    bool background_set = false;
    cv::Mat background_img;
};

#endif	/* BACKGROUNDSUBTRACTOR_H */

