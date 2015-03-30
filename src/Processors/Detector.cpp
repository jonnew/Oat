/* 
 * File:   Tracker.cpp
 * Author: Jon Newman <jpnewman snail mit dot edu>
 * 
 * Created on March 27, 2015, 1:40 PM
 */

#include "Detector.h"

#include <iostream>
#include <limits>
#include <math.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

Detector::Detector() {

    // Initialize area parameters without constraint
    min_object_area = 0;
    max_object_area = std::numeric_limits<double>::max();
    
    // Maximum number of contours defining candidate objects
    max_num_contours = 50;
    
    // Initial point is unknown
    object_found = false;
    xy_coord_px.x = 0;
    xy_coord_px.y = 0;
}

Detector::Detector(const Detector& orig) { }

Detector::~Detector() { }

bool Detector::findObjects(const cv::Mat& threshold_img) {

    cv::Mat thesh_cpy = threshold_img.clone();
    std::vector< std::vector < cv::Point > > contours;
    std::vector< cv::Vec4i > hierarchy;

    // This function will modify the threshold_img data.
    cv::findContours(thesh_cpy, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

    object_area = 0;
    object_found = false;

    if (int num_contours = hierarchy.size() > 0) {

        if (num_contours < max_num_contours) {

            for (int index = 0; index >= 0; index = hierarchy[index][0]) {

                cv::Moments moment = moments((cv::Mat)contours[index]);
                double area = moment.m00;

                // Isolate the largest contour within the min/max range.
                if (area > min_object_area && area < max_object_area && area > object_area) {
                    xy_coord_px.x = moment.m10 / area;
                    xy_coord_px.y = moment.m01 / area;
                    object_found = true;
                    object_area = area;
                }
            }
        }
        else {
            // Issue warning because we found too many contours
            status_text = "Too many contours. Tracking off.";

                    //std::cerr << "WARNING: Call to findObjects found more than the maximum allowed number of contours. " << std::endl;
                    //std::cerr << "Threshold image too noisy." << std::endl;
        }

    } else {
        // Issue warning because we found no countours
        status_text = "No contours. Tracking off.";
    }

    return object_found;
}

void Detector::decorateFeed(cv::Mat& display_img, const cv::Scalar& color) {
    
    // Add an image of the 
    if (object_found) {
        
        // Get the radius of the object
        int rad = sqrt(object_area/PI);
        cv::circle(display_img, xy_coord_px, rad, color, 2);
    }
    else {
        cv::putText(display_img, status_text, cv::Point(0, 50), 2, 1, cv::Scalar(0, 255, 0), 2);
    }
    
    
    
}
