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

#include <string>

#include <boost/program_options.hpp>

#include "../../lib/base/Component.h"
#include "../../lib/datatypes/Frame2.h"
#include "../../lib/datatypes/Pose.h"
#include "../../lib/shmemdf/Sink2.h"
#include "../../lib/shmemdf/Source.h"

namespace po = boost::program_options;

namespace oat {

class PositionDetector : public Component {
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

protected:
    /**
     * Perform object position detection.
     * @param Frame to look for object within.
     * @param pose Detected object pose.
     */
    virtual oat::Pose detectPose(oat::Frame &frame) = 0;

    // Detector name
    const std::string name_;

    // Check frame pixel color type
    // Default to not caring about color
    virtual bool checkPixelColor(oat::Pixel::Color c)
    {
        (void)c; // Override unused variable warning
        return true;
    }

    // Intrinsic parameters
    // cv::Matx33d camera_matrix_{cv::Matx33d::eye()};
    // std::vector<double> dist_coeff_{0, 0, 0, 0, 0, 0, 0, 0};

private:
    // Component Interface
    bool connectToNode(void) override;
    int process(void) override;

    // Frame source
    oat::FrameSource frame_source_;

    // Pose sink
    oat::Sink<oat::Pose> pose_sink_;
};

}      /* namespace oat */
#endif /* OAT_POSITIONDETECTOR_H */
