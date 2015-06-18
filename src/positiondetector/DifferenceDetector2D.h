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
    DifferenceDetector2D(const std::string& image_source_name, const std::string& position_sink_name);
    
    // Detector2D's public interface
    void configure(const std::string& config_file, const std::string& key);
    oat::Position2D detectPosition(cv::Mat& frame_in);

private:
    
    // Intermediate variables
    cv::Mat this_image, last_image;
    cv::Mat threshold_image;
    bool last_image_set;
    
    // Object detection
    double object_area;
    
    // The detected object position
    oat::Position2D object_position;
    
    // Detector parameters
    int difference_intensity_threshold;
    cv::Size blur_size;
    bool blur_on;
    
    // Tuning stuff
    bool tuning_windows_created;
    const std::string tuning_image_title;
    cv::Mat tune_image;
    void tune(void);
    void createTuningWindows(void);
    static void blurSliderChangedCallback(int, void*);

    // Processing segregation 
    // TODO: These are terrible - no IO sigature other than void -> void,
    // no exceptions, etc
    void applyThreshold(void);
    void set_blur_size(int value);
    void siftBlobs(void);
    void servePosition(void);

};

#endif	/* DIFFERENCEDETECTOR_H */

