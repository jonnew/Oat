/* 
 * File:   DifferenceDetector.h
 * Author: Jon Newman <jpnewman snail mit dot edu>
 *
 * Created on April 16, 2015, 6:27 PM
 */

#ifndef DIFFERENCEDETECTOR_H
#define	DIFFERENCEDETECTOR_H

#include "Detector.h"

#include "../../lib/shmem/MatClient.h"
#include "../../lib/shmem/SMServer.h"
#include "../../lib/shmem/Position2D.h"

class DifferenceDetector : public Detector{
public:
    DifferenceDetector();
    DifferenceDetector(const DifferenceDetector& orig);
    virtual ~DifferenceDetector();
    
    void findObject(void);
    void servePosition(void);
    
    void set_blur_size(int value);
    
private:
    
    // Intermediate variables
    cv::Mat last_image;
    cv::Mat threshold_image;
    bool last_image_set;
    
    // Object detection
    double object_area;
    shmem::Position2D object_position;
    
    // Detector parameters //TODO: config file input
    double threshold_level;
    double blur_size;
    bool blur_on;

    void thresholdImage(void);
    
    
    void createSliders(void);
    static void blurSliderChangedCallback(int, void*);
};

#endif	/* DIFFERENCEDETECTOR_H */

