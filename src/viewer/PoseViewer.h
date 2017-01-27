//******************************************************************************
//* File:   PoseViewer.h
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

#ifndef OAT_POSEVIEWER_H
#define OAT_POSEVIEWER_H

#include "Viewer.h"

#include <string>
#include <vector>

#include "../../lib/datatypes/Pose.h"

namespace oat {

class PoseViewer : public Viewer<oat::Pose> {
public:
    /**
     * @brief View a pose stream on the monitor.
     */
    using Viewer<oat::Pose>::Viewer;

private:
    // Implement ControllableComponent Interface
    void applyCommand(const std::string &command) override;
    oat::CommandDescription commands(void) override;

    // Implement Configurable Interface
    po::options_description options(void) const override;
    void applyConfiguration(const po::variables_map &vm,
                            const config::OptionTable &config_table) override;

    // Viewer Interface
    void display(const oat::Pose &pose) override;

    // Viewer initialized flag
    bool gui_inititalized_ {false};

    // Create image from pose
    cv::Mat generateFrame(const oat::Pose &pose) const;

    // Used to request a snapshot of the current image which is saved to disk
    std::atomic<bool> snapshot_requested_ {false};
    std::string snapshot_folder_;
    std::string snapshot_base_file_;
    void set_snapshot_path(const std::string &snapshot_path);
    void saveSnapshot(void);
};

}      /* namespace oat */
#endif /* OAT_POSEVIEWER_H */
