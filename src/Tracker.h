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
#include "Processors/HSVFilter.h"
#include "Processors/BackgroundSubtractor.h"
#include "Processors/Detector.h"
#include "Processors/Combiner.h"
#include "cpptoml.h"

class Tracker {
public:
    Tracker(std::string config_file);
    Tracker(const Tracker& orig);
    virtual ~Tracker();
    
    CameraControl cc;
    //BackgroundSubtractor br_subtractor;
    std::vector<HSVFilter> hsv_filters;
    std::vector<Detector> detectors;
    //Combiner combiner;
    
private:
    
    cv::Mat curr_image;
    std::vector<cv::Mat> filtered_images;
    
    cpptoml::table config;
    void build(void);

};

#endif	/* TRACKER_H */

