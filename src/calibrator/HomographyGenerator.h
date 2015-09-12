//******************************************************************************
//* File:   HomographyGenerator.h
//* Author: Jon Newman <jpnewman snail mit dot edu>
//*
//* Copyright (c) Jon Newman (jpnewman snail mit dot edu) 
//* All right reserved.
//* This file is part of the Oat project.
//* This is free software: you can redistribute it and/or modify
//* it under the terms of the GNU General Public License as published by
//* the Free Software Foundation, either version 3 of the License, or
//* (at your option) any later version.
//* This software is distributed in the hope that it will be useful,
//* but WITHOUT ANY WARRANTY; without even the implied warranty of
//* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//* GNU General Public License for more details.
//* You should have received a copy of the GNU General Public License
//* along with this source code.  If not, see <http://www.gnu.org/licenses/>.
//****************************************************************************

#ifndef HOMOGRAPHYGENERATOR_H
#define HOMOGRAPHYGENERATOR_H

#include <string>
#include <opencv2/core/mat.hpp>

#include "Calibrator.h"
#include "CalibratorVisitor.h"

/**
 * Interactive homography generator.
 */
class HomographyGenerator : public Calibrator {

    using point_size_t = std::vector<cv::Point2f>::size_type;

public:
    
    // Homography estimation procedure
    enum class EstimationMethod { ROBUST = 0, REGULAR, EXACT };

    /**
     * Interactive homography transform generator.  The user is presented with
     * a video display of the frame stream. The user then select points on the
     * video feed and enter their equivalent world-unit equivalent. Upon each
     * selection, the a best-fit homography matrix relating pixels to work
     * coordinates is calulated and the MSE between the transformed and
     * user-supplied positions is displayed. 
     * @param frame_source_name imaging setup frame source name
     */
    HomographyGenerator(const std::string& frame_source_name, EstimationMethod method);

    /**
     * Configure calibration parameters.
     * @param config_file configuration file path
     * @param config_key configuration key
     */
    void configure(const std::string& config_file, const std::string& config_key) override;
    
    void accept(std::uniqe_ptr<CalibratorVisitor> visitor);

protected:

    /**
     * Perform homography matrix generation routine.
     * @param frame current frame to use for running calibration
     */
    void calibrate(cv::Mat& frame) override;

private:

    // Is homography well-defined?
    bool homography_valid_;
    cv::Mat homography_;

    // Default esimation method
    EstimationMethod method_ {EstimationMethod::ROBUST};

    // Data used to create homography    
    std::vector<cv::Point2f> pixels_;
//    {
//                cv::Point2f(678, 349), 
//                cv::Point2f(672, 25), 
//                cv::Point2f(687, 682),
//                cv::Point2f(352, 364),
//                cv::Point2f(1010, 353)
//    };


    std::vector<cv::Point2f> world_points_;
//    {
//                cv::Point2f(0, 0), 
//                cv::Point2f(0, 1), 
//                cv::Point2f(0, -1),
//                cv::Point2f(-1, 0),
//                cv::Point2f(1, 0)
//    };
    
    // Current mouse point
    bool clicked_ {false};
    cv::Point mouse_pt_;

    // Methods for interactive session to obtain data points related pixel
    // coordinates and world coordinates, generate a homography, and display
    // the resuts
    int addDataPoint(void);
    int removeDataPoint(void); 
    void printDataPoints(std::ostream&);
    void printUsage(std::ostream&);
    int selectHomographyMethod(void);
    int generateHomography(void);
    int changeSavePath(void);
    int saveHomography(void);
    cv::Mat drawMousePoint(cv::Mat& frame);
    void onMouseEvent(int event, int x, int y);
    static void onMouseEvent(int event, int x, int y, int, void* _this);
    
};

#endif //HOMOGRAPHYGENERATOR_H
