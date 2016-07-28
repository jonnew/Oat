//******************************************************************************
//* Copyright (c) Jon Newman (jpnewman at mit snail edu)
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

#ifndef OAT_POSITION3D_H
#define	OAT_POSITION3D_H

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

#include "Position.h"

namespace oat {

using Point3D = cv::Point3d;
using Velocity3D = cv::Point3d;
using UnitVector3D = cv::Point3d;

class Position3D : public Position {

public:
    explicit Position2D(const std::string &label) :
      Position(label)
    {
        // Nothing
    }

    // 2D position primatives
    Point3D position;
    Velocity3D velocity;
    UnitVector3D heading;

    /**
     * @brief JSON Serializer
     *
     * @param writer Writer to use for serialization
     * @param verbose If true, specifies that fields be serialized even though
     * they contain indeterminate data? This is useful for ease of sample
     * alignment during post processing of saved files.
     */
    // TODO: Should this just return a rapidjson::Document which is redily
    // transformed into other formats such as msgpack, etc, instead of doing the 
    // writing right here?
    template <typename Writer>
    void Serialize(Writer& writer, bool verbose = false) const {

        writer.StartObject();

        // Sample number
        writer.String("tick");
        writer.Int(sample_.count());

        writer.String("usec");
        writer.Int64(sample_.microseconds().count());

        // Coordinate system
        writer.String("unit");
        writer.Int(static_cast<int>(unit_of_length_));

        // Position
        writer.String("pos_ok");
        writer.Bool(position_valid || verbose);

        if (position_valid || verbose) {
            writer.String("pos");
            writer.StartArray();
            writer.Double(position.x);
            writer.Double(position.y);
            writer.Double(position.z);
            writer.EndArray(2);
        }

        // Velocity
        writer.String("vel_ok");
        writer.Bool(velocity_valid || verbose);

        if (velocity_valid || verbose) {
            writer.String("vel");
            writer.StartArray();
            writer.Double(velocity.x);
            writer.Double(velocity.y);
            writer.Double(velocity.z);
            writer.EndArray(2);
        }

        // Head direction
        writer.String("head_ok");
        writer.Bool(heading_valid);

        if (heading_valid || verbose) {
            writer.String("head");
            writer.StartArray();
            writer.Double(heading.x);
            writer.Double(heading.y);
            writer.Double(heading.z);
            writer.EndArray(2);
        }

        // Categorical region 
        writer.String("reg_ok");
        writer.Bool(region_valid);

        if (region_valid || verbose) {
            writer.String("reg");
            writer.String(region);
        }

        writer.EndObject();
    }

    void setCoordSystem(const DistanceUnit value,
                        const cv::Matx33d homography) {
        unit_of_length_ = value;
        homography_ = homography;
    }

    cv::Matx33d homography() const { return homography_; }

private:

    cv::Matx33d homography_ {1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0};
};

}      /* namespace datatypes */
#endif /* OAT_POSITION3D_H */
