/* 
 * File:   Tracker.h
 * Author: Jon Newman <jpnewman snail mit dot edu>
 *
 * Created on March 29, 2015, 3:43 PM
 */

#ifndef TRACKER_H
#define	TRACKER_H

#include <string>
#include <opencv2/core/mat.hpp>

#include "CameraControl/CameraControl.h"
#include "Processors/HSVDetector.h"
#include "Processors/BackgroundSubtractor.h"
#include "Processors/Combiner.h"
#include "cpptoml.h"

class Tracker {
public:
    Tracker(std::string config_file);
    Tracker(const Tracker& orig);
    virtual ~Tracker();
    
    int run(void);
    
    CameraControl cc;
    BackgroundSubtractor subtractor;
    std::vector<HSVDetector> hsv_detectors;
    //Combiner combiner;
    
private:
    
    bool background_subtract_on = false;
    
    cv::Mat orig_image;
    cv::Mat proc_image;
    std::vector<cv::Mat> filtered_images;
    
    cpptoml::table config;
    void build(void);
    

};

#endif	/* TRACKER_H */

