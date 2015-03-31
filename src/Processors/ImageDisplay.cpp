/* 
 * File:   ImageDisplay.cpp
 * Author: Jon Newman <jpnewman snail mit dot edu>
 * 
 * Created on March 30, 2015, 3:50 PM
 */

#include "ImageDisplay.h"

#include <string>
#include <opencv2/highgui/highgui.hpp>  // Video write

void ImageDisplay::showImage() {
    cv::imshow(title, image);
}

void ImageDisplay::showImage(const cv::Mat& supplied_image) {
    cv::imshow(get_title(), supplied_image);
}

void ImageDisplay::showImage(std::string title_in, const cv::Mat& supplied_image) {
    cv::imshow(title_in, supplied_image);
}
