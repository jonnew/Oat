//******************************************************************************
//* File:   PositionDetector.h
//* Author: Jon Newman <jpnewman snail mit dot edu>
//
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

#ifndef OAT_POSITIONDETECTOR_H
#define	OAT_POSITIONDETECTOR_H

#define OAT_POSIDET_MAX_OBJ_AREA_PIX 100000
#define TUNE tuner_.registerParameter

#include "Tuner.h"

#include <string>

#include <boost/program_options.hpp>

#include "../../lib/base/Component.h"
#include "../../lib/base/Configurable.h"
#include "../../lib/datatypes/Frame.h"
#include "../../lib/datatypes/Pose.h"
#include "../../lib/shmemdf/Sink.h"
#include "../../lib/shmemdf/Source.h"

namespace po = boost::program_options;

namespace oat {

// Forward decl.
class SharedFrameHeader;

class PositionDetector : public Component, public Configurable<false> {
public:
    /**
     * Abstract object pose detector.
     * All concrete object position detector types implement this ABC.
     * @param frame_source_address Frame SOURCE node address
     * @param pose_sink_address Pose SINK node address
     */
    PositionDetector(const std::string &frame_source_address,
                     const std::string &pose_sink_address);
    virtual ~PositionDetector() { }

    // Component Interface
    oat::ComponentType type(void) const override { return oat::positiondetector; };
    std::string name(void) const override { return name_; }

protected:
    /**
     * Perform object position detection.
     * @param Frame to look for object within.
     * @param pose Detected object pose.
     */
    virtual void detectPosition(oat::Frame &frame, oat::Pose &pose) = 0;

    // Detector name
    const std::string name_;

    // Explicit frame data type
    oat::PixelColor required_color_ {PIX_ANY};

    // Tuning frame (shown in tuning window when tuning_on_ is true);
    bool tuning_on_ {false};
    oat::Frame tuning_frame_;
    Tuner tuner_;
    
    // Intrinsic parameters
    cv::Matx33d camera_matrix_ {cv::Matx33d::eye()};
    std::vector<double> dist_coeff_  {0, 0, 0, 0, 0, 0, 0, 0};

private:
    // Component Interface
    virtual bool connectToNode(void) override;
    int process(void) override;

    // Current pose
    oat::Pose * shared_pose_;

    // Frame source
    const std::string frame_source_address_;
    oat::Source<oat::Frame> frame_source_;

    // Pose sink
    const std::string pose_sink_address_;
    oat::Sink<oat::Pose> pose_sink_;
};

}      /* namespace oat */
#endif /* OAT_POSITIONDETECTOR_H */
