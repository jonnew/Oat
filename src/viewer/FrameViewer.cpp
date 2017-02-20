//******************************************************************************
//* File:   FrameViewer.cpp
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
//******************************************************************************

#include "FrameViewer.h"

#include <string>

#include <boost/filesystem.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/cvconfig.h>
#include <opencv2/highgui/highgui.hpp>

#include "../../lib/utility/FileFormat.h"
#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/ProgramOptions.h"
#include "../../lib/utility/TOMLSanitize.h"

namespace oat {

namespace bfs = boost::filesystem;
using msec = std::chrono::milliseconds;

po::options_description FrameViewer::options(void) const
{
    po::options_description local_opts;
    local_opts.add_options()
        ("display-rate,r", po::value<double>(),
         "Maximum rate, in Hz, at which the viewer is updated irrespective of "
         "its source's rate. If frames are supplied faster than this rate, "
         "they are ignored. Setting this to a reasonably low value prevents "
         "the viewer from consuming processing resources in order to update "
         "the display faster than is visually perceptible. Defaults to 30.")
        ("min-max,m", po::value<std::string>(),
         "2-element array of floats, [min,max], specifying the requested "
         "dyanmic range of the display. Pixel values below min will be mapped to "
         "min. Pixel values above max will be mapped to max. Others will be "
         "interprolated between min and max. Defaults to off.")
        ("snapshot-path,f", po::value<std::string>(),
         "The path to which in which snapshots will be saved. "
         "If a folder is designated, the base file name will be SOURCE. "
         "The time stamp of the snapshot will be prepended to the file name. "
         "Defaults to the current directory.")
        ;

    return local_opts;
}

void FrameViewer::applyConfiguration(const po::variables_map &vm,
                                     const config::OptionTable &config_table)
{
    // Display rate
    double r;
    if (oat::config::getNumericValue<double>(
            vm, config_table, "display-rate", r, 0.001)) {
        min_update_period_ms = Milliseconds(static_cast<int>(1000.0 / r));
    }

    // Min/max
    std::vector<double> m;
    if (oat::config::getArray<double, 2>(vm, config_table, "min-max", m, 0)) {

        auto min = m[0];
        auto max = m[1];

        if (min >= max || max > 255)
            throw std::runtime_error("Values of min-max should be between 0 "
                                     "and 255 and min must be less than max.");

        auto d = 255.0 / (max - min);
        for (size_t i = min; i < 256; i++) {
            if (i < max)
                lut_(i) = static_cast<unsigned char>((i-min) * d);
            else
                lut_(i) = 255;
        }

        min_max_defined_ = true;
    }

    // Snapshot save path
    std::string snapshot_path = "./";
    oat::config::getValue(vm, config_table, "snapshot-path", snapshot_path);
    set_snapshot_path(snapshot_path);
}

oat::CommandDescription FrameViewer::commands()
{
    const oat::CommandDescription commands{
        {"snap", "Take a snapshot of the current frame and save to the "
                 "configured snapshot path."}
    };

    return commands;
}

void FrameViewer::applyCommand(const std::string &command)
{
    const auto cmds = commands();
    if (command == "snap") {
        snapshot_requested_ = true;
    }
}

void FrameViewer::display(const oat::Frame &frame)
{
    // NOTE: This initialization is done here to ensure it is done by the same
    // thread that actually calls imshow(). If done in in the constructor, it
    // will not play nice with OpenGL.
    if (!gui_inititalized_) {
#ifdef HAVE_OPENGL
        try {
            cv::namedWindow(name_, cv::WINDOW_OPENGL & cv::WINDOW_KEEPRATIO);
        } catch (cv::Exception& ex) {
            oat::whoWarn(name_, "OpenCV not compiled with OpenGL support. "
                    "Falling back to OpenCV's display driver.\n");
            cv::namedWindow(name_, cv::WINDOW_NORMAL & cv::WINDOW_KEEPRATIO);
        }
#else
        cv::namedWindow(name_, cv::WINDOW_NORMAL & cv::WINDOW_KEEPRATIO);
#endif
        gui_inititalized_ = true;
    }

    if (frame.rows == 0 || frame.cols == 0)
        return;

    if (min_max_defined_)
        cv::LUT(frame, lut_, frame);

    cv::imshow(name_, frame);
    char gui_input = cv::waitKey(1);

    if (snapshot_requested_ || gui_input == 's') {

        // Reset snapshot request state
        snapshot_requested_ = false;

        // Generate current snapshot save path
        std::string fid;
        std::string timestamp = oat::createTimeStamp();

        int err = oat::createSavePath(fid,
                snapshot_folder_,
                snapshot_base_file_ + ".png",
                timestamp + "_",
                true);

        if (!err) {
            cv::imwrite(fid, frame);
            std::cout << "Snapshot saved to " << fid << "\n";
        } else {
            std::cerr << oat::Error("Snapshot file creation exited "
                    "with error " + std::to_string(err) + "\n");
        }
    }
}

void FrameViewer::set_snapshot_path(const std::string &snapshot_path)
{
    bfs::path path(snapshot_path.c_str());

    // Check that the snapshot save folder is valid
    if (!bfs::exists(path.parent_path())) {
        throw std::runtime_error("Requested snapshot save directory "
                                 "does not exist.\n");
    }

    // Get folder from path
    if (bfs::is_directory(path)) {
        snapshot_folder_ = path.string();
        snapshot_base_file_ = source_address_;
    } else {
        snapshot_folder_ = path.parent_path().string();

        // Generate base file name
        snapshot_base_file_ = path.stem().string();
        if (snapshot_base_file_.empty() || snapshot_base_file_ == ".")
            snapshot_base_file_ = source_address_;
    }
}

} /* namespace oat */
