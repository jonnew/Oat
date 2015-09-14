//******************************************************************************
//* File:   Calibrator.cpp
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

#include <memory>
#include <string>
#include <boost/filesystem.hpp>
#include <opencv2/core/mat.hpp>

#include "Calibrator.h"

namespace bfs = boost::filesystem;

bool Calibrator::process(void) {

    // Only proceed with processing if we are getting a valid frame
    if (frame_source_.getSharedMat(current_frame_)) {

        // Use the current frame for calibration
        calibrate(current_frame_);
    }

    // Check for end of frame stream
    return (frame_source_.getSourceRunState() == oat::ServerRunState::END);
}

bool Calibrator::generateSavePath(const std::string& save_path) {

    // Create folder and file name
    bfs::path path(save_path.c_str());
    std::string folder, file_name;

    if (bfs::is_directory(path)) {

        // Use default file name
        folder = path.string();
        file_name = "calibration";

    } else {

        // Use user-specified file name
        folder = path.parent_path().string();
        file_name = path.stem().string();
    }

    // Check that save directory is valid
    if (!bfs::exists(bfs::path(folder.c_str())))
        throw (std::runtime_error(
            "Requested calibration save path " + folder + " does not exist."));

    // Generate file name for this configuration
    calibration_save_path_ = folder + "/" + file_name + ".toml";

    // Check if file already exists
    return bfs::exists(calibration_save_path_.c_str());
}
