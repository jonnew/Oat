/* 
 * File:   PositionMeasurement.h
 * Author: Jon Newman <jpnewman snail mit dot edu>
 *
 * Created on March 29, 2015, 3:30 PM
 */

#ifndef POSITIONMEASUREMENT_H
#define	POSITIONMEASUREMENT_H

#include <opencv2/core/mat.hpp>

class PositionMeasurement {
public:
    PositionMeasurement();
    PositionMeasurement(const PositionMeasurement& orig);
    virtual ~PositionMeasurement();
    
private:
    
    cv::Point xy_coord_px;
    cv::Point xy_coord_mm;
    
};

class HeadPositionMeasurement : public PositionMeasurement{
public:
    HeadPositionMeasurement();
    HeadPositionMeasurement(const HeadPositionMeasurement& orig);
    virtual ~HeadPositionMeasurement();
    
private:
    
    cv::Point xy_coord_px;
    cv::Point xy_coord_mm;
    cv::Point direction;
    
};

#endif	/* POSITIONMEASUREMENT_H */

