/* 
 * File:   DifferenceDetector.h
 * Author: Jon Newman <jpnewman snail mit dot edu>
 *
 * Created on April 16, 2015, 6:27 PM
 */

#ifndef DIFFERENCEDETECTOR_H
#define	DIFFERENCEDETECTOR_H

#include "Detector.h"

class DifferenceDetector : public Detector {
public:
    DifferenceDetector(std::string image_source_name, std::string position_sink_name);
    
    void findObjectAndServePosition(void);
    void servePosition(void);
    void configure(std::string file_name, std::string key);
    
    void set_blur_size(int value);
    
private:
    
    // Intermediate variables
    cv::Mat this_image, last_image;
    cv::Mat threshold_image;
    bool last_image_set;
    
    // Object detection
    double object_area;
    shmem::Position object_position;
    
    // Detector parameters
    int difference_intensity_threshold;
    cv::Size blur_size;
    bool blur_on;

    void applyThreshold(void);
    void siftBlobs(void);

    void tune(void);
    void createTuningWindows(void);
    static void blurSliderChangedCallback(int, void*);
};

#endif	/* DIFFERENCEDETECTOR_H */

