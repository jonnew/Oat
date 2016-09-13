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
#include <opencv2/cvconfig.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../../lib/utility/FileFormat.h"
#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/ProgramOptions.h"
#include "../../lib/utility/TOMLSanitize.h"

namespace oat {

namespace bfs = boost::filesystem;
using namespace boost::interprocess;
using msec = std::chrono::milliseconds;

FrameViewer::FrameViewer(const std::string &source_address)
: Viewer<oat::Frame>(source_address)
{
    // Nothing
}

void FrameViewer::appendOptions(po::options_description &opts)
{
    // Accepts a config file
    Viewer<oat::Frame>::appendOptions(opts);

    // Common program options
    po::options_description local_opts;
    opts.add_options()
        ("display-rate,r", po::value<double>(),
         "Maximum rate at which the viewer is updated irrespective of its "
         "source's rate. If frames are supplied faster than this rate, they are "
         "ignored. Setting this to a reasonably low value prevents the viewer "
         "from consuming processing resorces in order to update the "
         "display faster than is visually perceptable. Defaults to 30.")
        ("snapshot-path,f", po::value<std::string>(),
        "The path to which in which snapshots will be saved. "
        "If a folder is designated, the base file name will be SOURCE. "
        "The timestamp of the snapshot will be prepended to the file name. "
        "Defaults to the current directory.")
        ;

    opts.add(local_opts);

    // Return valid keys
    for (auto &o: local_opts.options())
        config_keys_.push_back(o->long_name());
}

void FrameViewer::configure(const po::variables_map &vm)
{
    // Check for config file and entry correctness
    auto config_table = oat::config::getConfigTable(vm);
    oat::config::checkKeys(config_keys_, config_table);

    // Display rate
    double r;
    if (oat::config::getNumericValue<double>(
            vm, config_table, "display-rate", r, 0.001)) {
        min_update_period_ms = Milliseconds(static_cast<int>(1000.0 / r));
    }

    // Snapshot save path
    std::string snapshot_path = "./";
    oat::config::getValue(vm, config_table, "snapshot-path", snapshot_path);
    set_snapshot_path(snapshot_path);
}

void FrameViewer::display(const oat::Frame &frame)
{
    // NOTE: This inititalization is done here to enesure it is done by the same
    // thread that actually calls imshow(). If done in in the constructor, it will
    // not play nice with OpenGL.
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

    cv::imshow(name_, frame);
    char command = cv::waitKey(1);

    if (command == 's') {

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
            std::cerr << oat::Error("Snapshop file creation exited "
                    "with error " + std::to_string(err) + "\n");
        }
    }
}

void FrameViewer::set_snapshot_path(const std::string &snapshot_path)
{
    bfs::path path(snapshot_path.c_str());

    // Check that the snapshot save folder is valid
    if (!bfs::exists(path.parent_path())) {
        throw (std::runtime_error ("Requested snapshot save directory "
                                   "does not exist.\n"));
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
