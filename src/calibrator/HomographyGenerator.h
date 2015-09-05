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
#include <opencv2/core/matx.hpp>

#include "Calibrator.h"

/**
 * Interactive homography generator.
 */
class HomographyGenerator : public Calibrator {

    using point_size_t = std::vector<cv::Point2f>::size_type;

public:

    /**
     * Interactive homography transform generator.  The user is presented with
     * a video display of the frame stream. The user then select points on the
     * video feed and enter their equivalent world-unit equivalent. Upon each
     * selection, the a best-fit homography matrix relating pixels to work
     * coordinates is calulated and the MSE between the transformed and
     * user-supplied positions is displayed. 
     * @param frame_source_name imaging setup frame source name
     */
    HomographyGenerator(const std::string& frame_source_name);

    /**
     * Configure calibration parameters.
     * @param config_file configuration file path
     * @param config_key configuration key
     */
    void configure(const std::string& config_file, const std::string& config_key) override;

protected:

    /**
     * Perform homography matrix generation routine.
     * @param frame current frame to use for running calibration
     */
    void calibrate(cv::Mat& frame) override;

private:

    // User-supplied world unit definition. Defaults to meters.
    std::string world_units_;

    // Is homography well-defined?
    bool homography_valid_;
    cv::Mat homography_;
   
    // Data used to create homography    
    std::vector<cv::Point2f> pixels_ 
    {
                cv::Point2f(678, 349), 
                cv::Point2f(672, 25), 
                cv::Point2f(687, 682),
                cv::Point2f(352, 364),
                cv::Point2f(1010, 353)
    };


    std::vector<cv::Point2f> world_points_
    {
                cv::Point2f(0, 0), 
                cv::Point2f(0, 1), 
                cv::Point2f(0, -1),
                cv::Point2f(-1, 0),
                cv::Point2f(1, 0)
    };
    
    // Current mouse point
    bool clicked_ {false};
    cv::Point mouse_pt_;

    // Interactive session to 
    // (1) obtain pixel <-> world map
    // (2) Manipulate the data
    // (3) Display data and transform information
    // (4) Generate homography
    int addDataPoint(void);
    int removeDataPoint(void); 
    void printDataPoints(void);
    int generateHomography(void);
    int saveHomography(void);
    cv::Mat drawMousePoint(cv::Mat& frame);
    void onMouseEvent(int event, int x, int y);
    static void onMouseEvent(int event, int x, int y, int, void* _this);
    
};

#endif //HOMOGRAPHYGENERATOR_H