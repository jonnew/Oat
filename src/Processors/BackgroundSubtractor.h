/* 
 * File:   BackgroundSubtractor.h
 * Author: Jon Newman <jpnewman snail mit dot edu>
 *
 * Created on March 26, 2015, 6:08 PM
 */

#ifndef BACKGROUNDSUBTRACTOR_H
#define	BACKGROUNDSUBTRACTOR_H

#include <opencv2/core/mat.hpp>

class BackgroundSubtractor {
public:
    BackgroundSubtractor();
    BackgroundSubtractor(const BackgroundSubtractor& orig);
    virtual ~BackgroundSubtractor();
    
    void setBackgroundImage(const cv::Mat& input_img);
    void subtrackBackground(cv::Mat& input_img, cv::Mat& output_img);
    
    
private:
    
    // The background image used for subtraction
    cv::Mat background_img;
    
};

#endif	/* BACKGROUNDSUBTRACTOR_H */

