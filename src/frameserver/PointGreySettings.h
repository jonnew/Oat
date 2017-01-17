//******************************************************************************
//* File:   PointGreySettings.h
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

#ifndef OAT_POINTGREYSETTINGS_H
#define OAT_POINTGREYSETTINGS_H

#include <stdexcept>
#include <string>

#include "FlyCapture2.h"

namespace oat {

namespace pg = FlyCapture2;
using rte = std::runtime_error;

inline std::string pgPropertyStr(pg::PropertyType property) { return std::to_string(property); }

template <typename T>
void setPGAuto(T &cam, pg::PropertyType property)
{
    pg::Error error;
    pg::Property cam_prop;
    cam_prop.type = property;

    error = cam.GetProperty(&cam_prop);
    if (error != pg::PGRERROR_OK)
        throw rte("Error retrieving camera property "
                  + pgPropertyStr(property));

    cam_prop.autoManualMode = true;

    error = cam.SetProperty(&cam_prop, false);
    if (error != pg::PGRERROR_OK)
        throw rte("Error setting camera property " + pgPropertyStr(property));
}

template <typename T>
void setPGOff(T &cam, pg::PropertyType property)
{
    pg::Error error;
    pg::Property cam_prop;
    cam_prop.type = property;

    error = cam.GetProperty(&cam_prop);
    if (error != pg::PGRERROR_OK)
        throw rte("Error retrieving camera property "
                  + pgPropertyStr(property));

    cam_prop.onOff = false;

    error = cam.SetProperty(&cam_prop, false);
    if (error != pg::PGRERROR_OK)
        throw rte("Error setting camera property " + pgPropertyStr(property));
}

template <typename T>
float getPGAbsValue(T &cam, pg::PropertyType property) {

    pg::Error error;
    pg::Property cam_prop;

    cam_prop.type = property;

    error = cam.GetProperty(&cam_prop);
    if (error != pg::PGRERROR_OK)
        throw rte("Error retrieving camera property "
                  + pgPropertyStr(property));

    return cam_prop.absValue;
}

template <typename T>
float setPGAbsValue(T &cam, pg::PropertyType property, double val)
{
    auto value = static_cast<float>(val);

    pg::Error error;
    pg::Property cam_prop;
    pg::PropertyInfo cam_prop_info;

    cam_prop.type = property;
    cam_prop_info.type = property;

    error = cam.GetProperty(&cam_prop);
    if (error != pg::PGRERROR_OK)
        throw rte("Error retrieving camera property "
                  + pgPropertyStr(property));

    error = cam.GetPropertyInfo(&cam_prop_info);
    if (error != pg::PGRERROR_OK)
        throw rte("Error retrieving property info for "
                  + pgPropertyStr(property));

    if (!cam_prop_info.absValSupported)
        throw rte("Absolute value is not supported for property "
                  + pgPropertyStr(property));

    // Clamp the abs value (abs is somehow equiv to floating point)
    if (value > cam_prop_info.absMax)
        value = cam_prop_info.absMax;

    if (value < cam_prop_info.absMin)
        value = cam_prop_info.absMin;

    // Taken straight from Flycap source..
    if (property == pg::BRIGHTNESS) {
        // The brightness abs register sometimes starts drifting
        // due to a rounding error between the camera and the
        // actual value being held by the adjustment. To prevent
        // this, only apply the change to the camera if the
        // difference is greater than a specified amount.

        // Check if the difference is greater than 0.005f.
        const float difference = value - cam_prop.absValue;
        if (difference <= -0.005f || difference >= 0.005f) {
            // The difference is too small, don't do anything
            return value;
        }
    }

    cam_prop.absValue = value;
    cam_prop.absControl = true;
    cam_prop.autoManualMode = false;

    error = cam.SetProperty(&cam_prop, false);
    if (error != pg::PGRERROR_OK)
        throw rte("Error setting camera property " + pgPropertyStr(property));

    return value;
}

template <typename T>
std::vector<unsigned int> setPGValue(T &cam,
                                     pg::PropertyType property,
                                     unsigned int val_a,
                                     unsigned int val_b)
{
    auto values = std::vector<unsigned int>{val_a, val_b};

    // Data structs for camera settings update
    pg::Error error;
    pg::Property cam_prop;
    pg::PropertyInfo cam_prop_info;

    cam_prop.type = property;
    cam_prop_info.type = property;

    error = cam.GetProperty(&cam_prop);
    if (error != pg::PGRERROR_OK)
        throw rte("Error retrieving camera property "
                  + pgPropertyStr(property));

    error = cam.GetPropertyInfo(&cam_prop_info);
    if (error != pg::PGRERROR_OK)
        throw rte("Error retrieving property info for "
                  + pgPropertyStr(property));

    // Clamp the reg value (reg value is somehow equiv to uint)
    for (auto &v : values) {
        if (v > cam_prop_info.max)
            v = cam_prop_info.max;

        if (v < cam_prop_info.min)
            v = cam_prop_info.min;
    }

    // White balance has two values
    if (property == pg::WHITE_BALANCE) {

        cam_prop.valueA = values[0];
        cam_prop.valueB = values[1];

    } else {

        cam_prop.valueA = values[0];
    }

    cam_prop.absControl = false;
    cam_prop.autoManualMode = false;

    error = cam.SetProperty(&cam_prop, false);
    if (error != pg::PGRERROR_OK)
        throw rte("Error setting camera property " + pgPropertyStr(property));

    return values;
}

}      /* namespace oat */
#endif /* OAT_POINTGREYSETTINGS_H */
