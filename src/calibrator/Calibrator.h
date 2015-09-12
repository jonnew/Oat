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
     * @param frame_source_name Frame SOURCE name
     */
    Calibrator(const std::string& frame_source_name) :
      name_("calibrate[" + frame_source_name + "]")
    , frame_source_(frame_source_name)
    {
        // Nothing
    }

    virtual ~Calibrator() { //Nothing }

    /**
     * Run the calibration routine on the frame SOURCE.
     * @return True if SOURCE signals EOF
     */
    bool process(void);

    /**
     * Configure calibration parameters.
     * @param config_file configuration file path
     * @param config_key configuration key
     */
    virtual void configure(const std::string& config_file, const std::string& config_key) = 0;

    /**
     * Create the calibration file path using a specified path.
     * @param save_path the path to save configuration data. If this path is a
     * folder, the calibration file will default to calibraiton.toml. 
     * @return True if the specified calibration file already exists.
     */
    virtual bool generateSavePath(const std::string& save_path);

    // Accept method for visitors
    virtual void accept(std::uniqe_ptr<CalibratorVisitor> visitor) = 0;

    // Accessors
    const std::string& name(void) const { return name_; }

protected:

    /** Perform calibration routine.
     * @param frame frame to use for generating calibration parameters
     */
    virtual void calibrate(cv::Mat& frame) = 0;

    std::string calibration_save_path_; //!< Calibration parameter save path

private:

    std::string name_; //!< Calibrator name
    cv::Mat current_frame_; //!< Current frame provided by SOURCE
    oat::MatClient frame_source_; //!< The calibrator frame SOURCE 
};

#endif // CALIBRATOR_H
