/* 
 * File:   DifferenceDetector.h
 * Author: Jon Newman <jpnewman snail mit dot edu>
 *
 * Created on April 16, 2015, 6:27 PM
 */

#ifndef DIFFERENCEDETECTOR_H
#define	DIFFERENCEDETECTOR_H

#include "Detector2D.h"

class DifferenceDetector2D : public Detector2D {
public:
    DifferenceDetector2D(std::string image_source_name, std::string position_sink_name);
    
    // Detector2D's public interface
    void configure(std::string file_name, std::string key);
    void findObjectAndServePosition(void);

private:
    
    // Intermediate variables
    cv::Mat this_image, last_image;
    cv::Mat threshold_image;
    bool last_image_set;
    
    // Object detection
    double object_area;
    
    // Detector parameters
    int difference_intensity_threshold;
    cv::Size blur_size;
    bool blur_on;
    
    
    void applyThreshold(void);
    void set_blur_size(int value);
    void siftBlobs(void);
    void servePosition(void);

    void tune(void);
    void createTuningWindows(void);
    
    static void blurSliderChangedCallback(int, void*);
};

#endif	/* DIFFERENCEDETECTOR_H */

