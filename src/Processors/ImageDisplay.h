/* 
 * File:   ImageDisplay.h
 * Author: Jon Newman <jpnewman snail mit dot edu>
 *
 * Created on March 30, 2015, 3:50 PM
 */

#ifndef IMAGEDISPLAY_H
#define	IMAGEDISPLAY_H

#include <string>
#include <opencv2/core/mat.hpp>

class ImageDisplay {
    
public:
    void showImage(void);
    void showImage(const cv::Mat&);
    void showImage(std::string title_in, const cv::Mat&);
    
    bool get_show(void) { return show; }
    std::string get_title(void) { return title; }
    cv::Mat get_image(void) { return image; }
    
protected:
    
    // Should the image be displayed?
    bool show = false;
    
    // Image title
    std::string title;
    
    // Image data
    cv::Mat image;
   
};

#endif	/* IMAGEDISPLAY_H */

