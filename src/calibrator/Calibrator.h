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
#include <boost/filesystem.hpp>
#include <opencv2/core/mat.hpp>

#include "../../lib/shmem/MatClient.h"

namespace bfs = boost::filesystem;

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
    Calibrator(const std::string& frame_source_name) :
      name_("calibrate[" + frame_source_name + "]")
    , frame_source_(frame_source_name) { 
      
      
    
      }

    virtual ~Calibrator() { }

    /**
     * Run the calibration routine on the frame SOURCE.
     */
    bool process(void) {

        // Only proceed with processing if we are getting a valid frame
        if (frame_source_.getSharedMat(current_frame_)) {

            // Use the current frame for calibration
            calibrate(current_frame_);
        }
        
        // Check for end of frame stream
        return (frame_source_.getSourceRunState() == oat::ServerRunState::END);
    }

    /**
     * Configure calibration parameters.
     * @param config_file configuration file path
     * @param config_key configuration key
     */
    virtual void configure(const std::string& config_file, const std::string& config_key) = 0;

    /**
     * Make the calibration file path.
     * return True if file already exists.
     */
    virtual bool generateSavePath(const std::string& save_path) {

        // Check that the save folder is valid
        bfs::path path(save_path.c_str());
        if (!bfs::exists(path.parent_path())) {
            throw (std::runtime_error("Requested calibration save path, " +
                    save_path + ", does not exist.\n"));
        }

        std::string file_name_ = path.stem().string();
        if (file_name_.empty())
            file_name_ = "calibration";

        // Generate file name for this configuration
        std::string folder = bfs::path(save_path.c_str()).parent_path().string();
        calibration_save_path_ = folder + "/" + file_name_ + ".toml";


       
        return bfs::exists(calibration_save_path_.c_str());
    }
    
    // Accessors
    std::string name(void) const { return name_; }

protected:

    /**
     * Perform calibration routine.
     * @param frame frame to use for generating calibration parameters
     */
    virtual void calibrate(cv::Mat& frame) = 0;
    
    // Path to save calibration parameters 
    std::string calibration_save_path_;
    
    
private:

    // Viewer name
    std::string name_;

    // Current frame provided by SOURCE
    cv::Mat current_frame_;

    // Frame SOURCE to get frames for calibration
    oat::MatClient frame_source_;
};

#endif //CALIBRATOR_H