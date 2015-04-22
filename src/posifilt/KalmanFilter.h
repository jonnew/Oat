/* 
 * File:   KalmanFilter.h
 * Author: Jon Newman <jpnewman snail mit dot edu>
 *
 * Created on April 21, 2015, 9:35 PM
 */

#ifndef KALMANFILTER_H
#define	KALMANFILTER_H

#include <opencv2/opencv.hpp>

#include "PositionFilter.h"

class KalmanFilter : public PositionFilter {
public:
    KalmanFilter(std::string position_source_name, std::string position_sink_name);

    void grabPosition(void);
    void filterPosition(void);
    void serveFilteredPosition(void);
    
    
private:

    shmem::Position raw_position;

    // Sample period
    float dt;
    
    // Standard deviation of assumed random accelerations.
    float sig_accel;
	float sig_measure_noise;
    
    cv::KalmanFilter kf;
    void configureFilter();
};

#endif	/* KALMANFILTER_H */

