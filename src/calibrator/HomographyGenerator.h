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
 * Interactive homography transform generator.
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
    void configure(const std::string& config_file, const std::string& config_key);

protected:

    /**
     * Perform homography matrix generation routine.
     * @param frame current frame to use for running calibration
     */
    void calibrate(cv::Mat& frame);

private:

    // User-supplied world unit definition. Defaults to meters.
    std::string world_units_;

    // Is homography well-defined?
    bool homography_valid_;
    cv::Matx33d homography_;
   
    // Data used to create homography
    bool added_ {false};
    std::vector<cv::Point2f> pixels_, world_points_;
    
    // Current mouse point
    bool clicked_ {false};
    cv::Point mouse_pt_;

    // Show frame and start interactive session
    //void showFrame(const cv::Mat& frame);
    void printDataPoints();
    cv::Mat addMousePoint(cv::Mat& frame);
    void onMouseEvent(int event, int x, int y);
    static void onMouseEvent(int event, int x, int y, int, void* _this);
    
    // Data list manipulation 
    void addDataPoint(const std::pair<cv::Point2f, cv::Point2f>&& new_point);
    void removeDataPoint(const point_size_t index_to_remove); 

    // Propossed methods 
    //void catchLeftClick();
    //void getPositionFromStdIO();
    void generateHomography(void);

};

#endif //HOMOGRAPHYGENERATOR_H