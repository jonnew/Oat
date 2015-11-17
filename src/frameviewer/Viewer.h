//******************************************************************************
//* File:   Viewer.h
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

#ifndef OAT_VIEWER_H
#define OAT_VIEWER_H

#include <chrono>
#include <string>
#include <opencv2/core/mat.hpp>

#include "../../lib/shmemdf/Source.h"

namespace oat {

// Forward decl.
class SharedCVMat;

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
    Viewer(const std::string &frame_source_name,
           const std::string &save_path);

    void connectToNode(void);
    bool showImage(void);
    void generateSnapshotPath(void);

    // Accessors
    inline std::string name() const { return name_; }

    // Constants
    static constexpr Milliseconds MIN_UPDATE_PERIOD_MS {33};
    static constexpr int COMPRESSION_LEVEL {9};

private:

    // Viewer name
    std::string name_;

    // Image data
    cv::Mat internal_frame_;

    // Frame SOURCE to get frames to display
    const std::string frame_source_address_;
    oat::NodeState node_state_;
    oat::Source<oat::SharedCVMat> frame_source_;

    // Minimum viewer refresh period
    Clock::time_point tick_, tock_;

    // Used to request a snapshot of the current image, saved to disk
    std::string snapshot_path_;
    std::string file_name_;
    std::vector<int> compression_params_;


    /**
     * Make the snapshot file path using the requested save folder
     * and current timestamp.
     * return Snapshot filepath
     */
    std::string makeFileName(void);
};

}      /* namespace oat */
#endif /* OAT_VIEWER_H */
