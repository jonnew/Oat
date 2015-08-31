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

#include "../../lib/shmem/MatClient.h"

#include "Calibrator.h"

/**
 * Interactive homography transform generator.
 */
class HomographyGenerator : public Calibrator {

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
    void configure(const std::string& config_file, const std::string& config_key) = 0;

    // Accessors
    std::string name(void) const { return name_; }

protected:

    /**
     * Perform homography matrix generation routine.
     * @param frame current frame to use for running calibration
     */
    void calibrate(cv::Mat& frame);

private:
    
    showFrame(const cv::Mat& frame);
    
    // Propossed methods 
    //void catchLeftClick();
    //void getPositionFromStdIO();
    //void generateHomography(std::vector<cv::Point<double> > pixels, 
    //        std::vector<cv::Point<double> > world_units);

};
