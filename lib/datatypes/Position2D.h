
//* File:   Position2D.h
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

#ifndef OAT_POSITION2D_H
#define	OAT_POSITION2D_H

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

#include <rapidjson/prettywriter.h>

#include "Position.h"

namespace oat {

using Point2D = cv::Point2d;
using Velocity2D = cv::Point2d;
using UnitVector2D = cv::Point2d;

class Position2D : public Position {

public:
    explicit Position2D(const std::string &label) :
      Position(label)
    {
        // Nothing
    }

    // 2D position primatives
    Point2D position;
    Velocity2D velocity;
    UnitVector2D heading;

    /**
     * @brief JSON Serializer
     *
     * @param writer Writer to use for serialization
     * @param verbose Should fields be serialed even though they contain
     * indeterminate data? This is useful for ease of sample alignment during
     * post processing of saved files.
     */
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
            writer.String("pos_xy");
            writer.StartArray();
            writer.Double(position.x);
            writer.Double(position.y);
            writer.EndArray(2);
        }

        // Velocity
        writer.String("vel_ok");
        writer.Bool(velocity_valid || verbose);

        if (velocity_valid || verbose) {
            writer.String("vel_xy");
            writer.StartArray();
            writer.Double(velocity.x);
            writer.Double(velocity.y);
            writer.EndArray(2);
        }

        // Head direction
        writer.String("head_ok");
        writer.Bool(heading_valid);

        if (heading_valid || verbose) {
            writer.String("head_xy");
            writer.StartArray();
            writer.Double(heading.x);
            writer.Double(heading.y);
            writer.EndArray(2);
        }

        // Head direction
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

}      /* namespace oat */
#endif /* OAT_POSITION2D_H */
