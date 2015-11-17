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

namespace oat {

    
Calibrator::Calibrator(const std::string &frame_source_address) :
  name_("calibrate[" + frame_source_address + "]")
, frame_source_address_(frame_source_address)
{
    // Nothing
}

void Calibrator::connectToNode() {

    frame_source_.connect(frame_source_address_);
}

bool Calibrator::process(void) {

    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sink to write to node
    node_state_ = frame_source_.wait();
    if (node_state_ == oat::NodeState::END)
        return true;

    // Clone the shared frame
    internal_frame_ = frame_source_.clone();

    // Tell sink it can continue
    frame_source_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //
    
    calibrate(internal_frame_);

    // Sink was not at END state 
    return false;
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

} /* namespace oat */
