/* 
 * File:   Combiner.h
 * Author: Jon Newman <jpnewman snail mit dot edu>
 *
 * Created on March 29, 2015, 3:17 PM
 */

#ifndef COMBINER_H
#define	COMBINER_H

#include <opencv2/core/mat.hpp>

class PositionMeasurement;
class Tracker;

class Combiner {
    
       friend Tracker;
protected:
    
    virtual PositionMeasurement calculatePosition(std::vector<cv::Point>& raw_coordinates);
    virtual PositionMeasurement filterPosition(PositionMeasurement input_pos); // TODO: Kalman

};

#endif	/* COMBINER_H */

