//******************************************************************************
//* File:   FrameViewer.h
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

#ifndef VIEWER_H
#define VIEWER_H

#include <chrono>
#include <string>
#include <opencv2/core/mat.hpp>

#include "../../lib/shmem/MatClient.h"

/**
 * View a frame stream on the monitor.
 */
class Viewer {
    
    using Clock = std::chrono::high_resolution_clock;
    using Milliseconds = std::chrono::milliseconds;
    
public:
    
    /**
     * View a frame stream on the monitor.
     */
    Viewer(const std::string& frame_source_name, 
           const std::string& save_path);

    bool showImage(void);
    bool showImage(std::string title);

    // Accessors
    std::string get_name(void) const { return name; }
    
private:

    // Viewer name
    std::string name;

    // Image data
    cv::Mat current_frame;

    // Frame SOURCE to get frames to display
    oat::MatClient frame_source;
    
    // Minimum viewer refresh period
    Clock::time_point tick, tock;
    const Milliseconds min_update_period {33};
    
    // Used to request a snapshot of the current image, saved to disk
    std::string snapshot_path_;
    std::string file_name;
    std::vector<int> compression_params;
    const int compression_level {9};

    /**
     * Make the snapshot file path using the requested save folder
     * and current timestamp.
     * return Snapshot filepath
     */
    std::string makeFileName(void);
};

#endif //VIEWER_H
