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

#include "Calibrator.h"

#include <string>
#include <boost/filesystem.hpp>
#include <opencv2/core/mat.hpp>

#include "../../lib/utility/TOMLSanitize.h"

namespace bfs = boost::filesystem;

namespace oat {

Calibrator::Calibrator(const std::string &source_address) :
  name_("calibrate[" + source_address + "]")
, source_address_(source_address)
{
    // Nothing
}

bool Calibrator::connectToNode()
{
    // Establish our a slot in the node
    frame_source_.touch(source_address_);

    // Wait for sychronous start with sink when it binds the node
    if (frame_source_.connect() != SourceState::CONNECTED)
        return false;

    return true;
}

int Calibrator::process(void)
{
    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sink to write to node
    if (frame_source_.wait() == oat::NodeState::END)
        return 1;

    // Clone the shared frame
    frame_source_.copyTo(internal_frame_);

    // Tell sink it can continue
    frame_source_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    calibrate(internal_frame_);

    // Sink was not at END state
    return 0;
}

// TODO: Replace with common utility?
bool Calibrator::generateSavePath(const std::string &save_path,
                                  const std::string &default_name)
{

    // Create folder and file name
    bfs::path path(save_path.c_str());
    std::string folder, file_name;

    if (bfs::is_directory(path)) {

        // Use default file name
        folder = path.string();
        file_name = default_name;

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
