//******************************************************************************
//* File:   Calibrator.h
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

#ifndef CALIBRATOR_H
#define CALIBRATOR_H

#include <string>
#include <opencv2/core/mat.hpp>

#include "../../lib/shmem/MatClient.h"

/**
 * Abstract calibrator.
 * All concrete calibrator types implement this ABC.
 */
class Calibrator {

public:

    /**
     * Abstract calibrator.
     * All concrete calibrator types implement this ABC.
     */
    Calibrator(const std::string& frame_source_name);

    virtual ~Calibrator() { }

    /**
     * Run the calibration routine on the frame SOURCE.
     */
    bool process(void) {

        // Only proceed with processing if we are getting a valid frame
        if (frame_source.getSharedMat(current_frame)) {

            // Use the current frame for calibration
            calibrate(current_frame);
        }
        
        // Check for end of frame stream
        return (frame_source.getSourceRunState() == oat::ServerRunState::END);
    }

    /**
     * Configure calibration parameters.
     * @param config_file configuration file path
     * @param config_key configuration key
     */
    virtual void configure(const std::string& config_file, const std::string& config_key) = 0;

    // Accessors
    std::string name(void) const { return name_; }

protected:

    /**
     * Perform calibration routine.
     * @param frame frame to use for generating calibration parameters
     */
    virtual void calibrate(cv::Mat& frame) = 0;

private:

    // Viewer name
    std::string name_;

    // Current frame provided by SOURCE
    cv::Mat current_frame_;

    // Frame SOURCE to get frames for calibration
    oat::MatClient frame_source_;

    // Path to save calibration parameters 
    std::string file_name_;
    std::string save_path_;
    const bool append_date_;

    /**
     * Make the calibration file path using the requested save folder
     * and current timestamp.
     * return Calibration file path
     */
    std::string makeFileName(void);
};
