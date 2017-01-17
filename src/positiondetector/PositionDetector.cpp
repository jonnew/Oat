//******************************************************************************
//* File:   PositionDetector.cpp
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

#include <string>
#include <opencv2/core/mat.hpp>

#include "../../lib/datatypes/Position2D.h"
#include "../../lib/shmemdf/Source.h"
#include "../../lib/shmemdf/Sink.h"

#include "PositionDetector.h"

namespace oat {

PositionDetector::PositionDetector(const std::string &frame_source_address,
                                   const std::string &pose_sink_address)
: name_("posidet[" + frame_source_address + "->" + pose_sink_address + "]")
, tuner_(name_)
, frame_source_address_(frame_source_address)
, pose_sink_address_(pose_sink_address)
{
  // Nothing
}

bool PositionDetector::connectToNode()
{
    // Establish our a slot in the node
    frame_source_.touch(frame_source_address_);

    // Wait for synchronous start with sink when it binds its node
    if (frame_source_.connect(required_color_) != SourceState::CONNECTED)
        return false;

    // Bind to sink node and create a shared position
    pose_sink_.bind(pose_sink_address_);
    shared_pose_ = pose_sink_.retrieve();

    return true;
}

int PositionDetector::process()
{
    oat::Frame internal_frame;
    oat::Pose pose;

    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sink to write to node
    if (frame_source_.wait() == oat::NodeState::END)
        return 1;

    // Clone the shared frame
    frame_source_.copyTo(internal_frame);

    // Tell sink it can continue
    frame_source_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    // Propagate sample info and detect position
    pose.set_sample(internal_frame.sample());
    detectPosition(internal_frame, pose);

    if (tuning_on_)
        tuner_.tune(tuning_frame_, pose, camera_matrix_, dist_coeff_);

    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sources to read
    pose_sink_.wait();

    *shared_pose_ = pose;

    // Tell sources there is new data
    pose_sink_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    // Sink was not at END state
    return 0;
}

//void PositionDetector::tune(const oat::Pose &pose)
//{
//    if (!tuning_windows_created_) {
//#ifdef HAVE_OPENGL
//        try {
//            cv::namedWindow(name_,
//                            cv::WINDOW_OPENGL & cv::WINDOW_KEEPRATIO);
//        } catch (cv::Exception &ex) {
//            whoWarn(name_, "OpenCV not compiled with OpenGL support. Falling "
//                           "back to OpenCV's display driver.\n");
//            cv::namedWindow(name_, cv::WINDOW_NORMAL & cv::WINDOW_KEEPRATIO);
//        }
//#else
//        cv::namedWindow(name_, cv::WINDOW_NORMAL);
//#endif
//        setupTuningParameters(name_);
//        tuning_windows_created_ = true;
//    }
//
//    std::string msg = cv::format("Not found");
//    if (pose.found) {
//
//        // Project using tvec and rvec
//        //std::vector<cv::Point3f> pts{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
//        //auto x = cv::projectPoints(pts, pose.r(), pose.t(),K,D);
//
//        // Show axes
//        //std::string type_s
//        //    = pose.unit_of_length == oat::Pose::DistanceUnit::Pixels ?
//        //          "pixels" :
//        //          "meters";
//        //auto xyz = pose.t<cv::Point3d>();
//        //msg = cv::format("(%d, %d, %d)", (int)xyz.x, (int)xyz.y,
//        //        (int)xyz.z);
//    }
//
//    int baseline = 0;
//    cv::Size textSize = cv::getTextSize(msg, 1, 1, 1, &baseline);
//    cv::Point text_origin(tuning_frame_.cols - textSize.width - 10,
//                          tuning_frame_.rows - 2 * baseline - 10);
//
//    cv::putText(tuning_frame_, msg, text_origin, 1, 1, cv::Scalar(0, 255, 0));
//    cv::imshow(name_, tuning_frame_);
//    cv::waitKey(1);
//}

} /* namespace oat */
