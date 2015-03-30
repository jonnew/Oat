/* 
 * File:   Tracker.h
 * Author: Jon Newman <jpnewman snail mit dot edu>
 *
 * Created on March 27, 2015, 1:40 PM
 */

#ifndef Tracker_H
#define Tracker_H
#define PI 3.141592653589793238462643383

#include <string>

#include <opencv2/core/mat.hpp>

class Detector {
    
    friend class Combiner;
    
public:

    Detector(void);
    Detector(const Detector& orig);
    virtual ~Detector();

    bool findObjects(const cv::Mat& threshold_img);
    void decorateFeed(cv::Mat& display_img, const cv::Scalar&);

    // TODO: Kalman2D (extendes generic Filter2D)

    // Accessors
    void set_max_num_contours(unsigned int max_num_contours) {
        max_num_contours = max_num_contours;
    }

    void set_min_object_area(double min_object_area) {
        min_object_area = min_object_area;
    }

    void set_max_object_area(double max_object_area) {
        max_object_area = max_object_area;
    }

private:

    float mm_per_px;

    std::string status_text;
    bool object_found;
    double object_area;
    cv::Point xy_coord_px;
    cv::Point xy_coord_mm;

    unsigned int max_num_contours;
    double min_object_area;
    double max_object_area;


};
#endif //Tracker_H



