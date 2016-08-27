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

#ifndef OAT_FRAMEVIEWER_H
#define OAT_FRAMEVIEWER_H

#include "Viewer.h"

#include <string>
#include <vector>

#include "../../lib/datatypes/Frame.h"

namespace oat {

/**
 * View a frame stream on the monitor.
 */
class FrameViewer : public Viewer<oat::Frame> {

public:

    FrameViewer(const std::string &source_name);

    /**
     * @brief Append type-specific program options.
     * @param opts Program option description to be specialized.
     */
    void appendOptions(po::options_description &opts) const override;

    /**
     * @brief Configure filter parameters.
     * @param vm Previously parsed program option value map.
     */
    void configure(const po::variables_map &vm) override;

private:

    bool gui_inititalized_ {false};

    // Used to request a snapshot of the current image, saved to disk
    std::string snapshot_folder_;
    std::string snapshot_base_file_;

    void set_snapshot_path(const std::string &snapshot_path);
    void saveSnapshot(void);
    void display(const oat::Frame &frame) override;
};

}      /* namespace oat */
#endif /* OAT_FRAMEVIEWER_H */
