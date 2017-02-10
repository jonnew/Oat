//******************************************************************************
//* File:   PoseViewer.cpp
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

#include "PoseViewer.h"

#include <iomanip>
#include <sstream>
#include <string>

#include <boost/filesystem.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../../lib/utility/FileFormat.h"
#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/ProgramOptions.h"
#include "../../lib/utility/TOMLSanitize.h"

namespace oat {

namespace bfs = boost::filesystem;
using msec = std::chrono::milliseconds;

po::options_description PoseViewer::options(void) const
{
    po::options_description local_opts;
    local_opts.add_options()
        ("display-rate,r", po::value<double>(),
         "Maximum rate, in Hz, at which the viewer is updated irrespective of "
         "its source's rate. If frames are supplied faster than this rate, "
         "they are ignored. Setting this to a reasonably low value prevents "
         "the viewer from consuming processing resources in order to update "
         "the display faster than is visually perceptible. Defaults to 30.")
        ("snapshot-path,f", po::value<std::string>(),
         "The path to which in which snapshots will be saved. "
         "If a folder is designated, the base file name will be SOURCE. "
         "The time stamp of the snapshot will be prepended to the file name. "
         "Defaults to the current directory.")
        ;

    return local_opts;
}

void PoseViewer::applyConfiguration(const po::variables_map &vm,
                                     const config::OptionTable &config_table)
{
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

oat::CommandDescription PoseViewer::commands()
{
    const oat::CommandDescription commands{
        {"snap", "Take a snapshot of the current frame and save to the "
                 "configured snapshot path."}
    };

    return commands;
}

void PoseViewer::applyCommand(const std::string &command)
{
    std::cout << command << "\n";
    const auto cmds = commands();
    if (command == "snap") {
        snapshot_requested_ = true;
    }
}

void PoseViewer::display(const oat::Pose &pose)
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

    auto frame = generateFrame(pose);
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

cv::Mat PoseViewer::generateFrame(const oat::Pose &pose) const
{
    if (pose.orientation_dof == Pose::DOF::Zero)
        throw std::runtime_error("No orientation information to view.");

    using PlygnVert = std::array<cv::Point3f, 3>;

    // Create mat to draw on
    const double frame_px = 500;
    cv::Mat frame = cv::Mat::zeros(
        {static_cast<int>(frame_px), static_cast<int>(frame_px)}, CV_8UC3);

    // Messages to be printed on screen
    std::vector<std::string> msgs;

    if (pose.found) {

        // Translate rotation matrix to frame center
        auto T = pose.orientation<cv::Matx44d>();
        T(0, 3) = frame_px / 2;
        T(1, 3) = frame_px / 2;

        // 3D axis
        const float l = 40;
        std::vector<cv::Point3f> axis_3d;
        axis_3d.push_back(cv::Point3f(l, 0, 0));
        axis_3d.push_back(cv::Point3f(0, l, 0));
        axis_3d.push_back(cv::Point3f(0, 0, l));
        axis_3d.push_back(cv::Point3f(0, 0, 0));
        cv::perspectiveTransform(axis_3d, axis_3d, T);

        // Sort by z-value
        std::vector<size_t> axis_idx{0, 1, 2};
        std::vector<cv::Scalar> axis_col{{0, 0, 255}, {0, 255, 0}, {255, 0, 0}};
        std::sort(axis_idx.begin(),
                  axis_idx.end(),
                  [&axis_3d](size_t a, size_t b) {
                      return axis_3d[a].z > axis_3d[b].z;
                  });

        const float off = 200;
        auto origin = cv::Point2f(axis_3d[3].x - off, axis_3d[3].y + off);
        for (const auto &i : axis_idx) {
            auto proj = cv::Point2f(axis_3d[i].x - off, axis_3d[i].y + off);
            cv::line(frame, origin, proj, axis_col[i], 3);
        }

        // Airplane
        const auto yellow = cv::Scalar(0, 255, 255);
        const auto white = cv::Scalar(255, 255, 255);
        const std::vector<cv::Point3f> ap{{100, 0, 0},
                                          {-100, -50, 0},
                                          {-100, -10, 0},
                                          {-100, 0, 30},
                                          {-100, 10, 0},
                                          {-100, 50, 0}};
        cv::perspectiveTransform(ap, ap, T);

        static constexpr size_t num_points = 4;
        std::array<PlygnVert, num_points> poly;
        poly[0] = {{ap[0], ap[1], ap[2]}};
        poly[1] = {{ap[0], ap[2], ap[3]}};
        poly[2] = {{ap[0], ap[3], ap[4]}};
        poly[3] = {{ap[0], ap[4], ap[5]}};

        // Sort polygons by maximal z-value of non-common verticies
        std::sort(poly.begin(), poly.end(), [](PlygnVert A, PlygnVert B) {
            auto ma = std::numeric_limits<float>::min();
            auto mb = std::numeric_limits<float>::min();

            // BRITTLE HACK! Relies on ordering of poly. 
            ma = A[1].z > A[2].z ? A[1].z : A[2].z;
            mb = B[1].z > B[2].z ? B[1].z : B[2].z;

            return mb > ma;
        });

        for (const auto &p : poly) {
            cv::Point x[num_points];
            for (size_t i = 0; i < p.size(); i++)
                x[i] = {static_cast<int>(p[i].x), static_cast<int>(p[i].y)};
            cv::fillConvexPoly(frame, x, 3, cv::Scalar(75, 75, 75));
            for (size_t i = 0; i < p.size(); i++) {
                auto j = (i+1) % p.size();
                cv::line(frame, x[i], x[j], white, 1);
                cv::circle(frame, x[i], 1, yellow, 2);
            }
        }

        // Position
        auto p = pose.position<std::array<double, 3>>();
        std::stringstream m;
        m << std::setprecision(2) << std::fixed;
        if (pose.unit_of_length == Pose::DistanceUnit::Meters)
            m << "P (m): [";
        else if (pose.unit_of_length == Pose::DistanceUnit::Pixels)
            m << "P (px): [";
        m << p[0] << ", " << p[1] << ", " << p[2] << "]";
        msgs.push_back(m.str());

        // Tait-bryan angles
        auto tb = pose.toTaitBryan(true);
        m.str("");
        m << " (deg): [" << tb[0] << ", " << tb[1] << ", " << tb[2] << "]";

        msgs.push_back(m.str());

    } else {
        msgs.push_back("Not found");
    }

    for (std::vector<std::string>::size_type i = 0; i < msgs.size(); i++) {

        int baseline = 0;
        cv::Size textSize = cv::getTextSize(msgs[i], 1, 1, 1, &baseline);
        cv::Point text_origin(frame.cols - textSize.width - 10,
                              frame.rows - (i * (textSize.height + 10)) - 10 );
        cv::putText(frame, msgs[i], text_origin, 1, 1, cv::Scalar(0, 255, 255));
    }

    return frame;
}

void PoseViewer::set_snapshot_path(const std::string &snapshot_path)
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

