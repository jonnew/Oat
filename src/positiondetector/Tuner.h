//******************************************************************************
//* File:   Tuner.h
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

#ifndef OAT_TUNER_H
#define	OAT_TUNER_H

#include <iostream>

#include <functional>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>

#include <opencv2/highgui.hpp>

#include "../../lib/datatypes/Frame.h"
#include "../../lib/datatypes/Pose.h"

namespace oat {

struct Tuner {

    struct CBBase {
        virtual ~CBBase() { } 
    };

    template <typename T>
    struct CallbackParams : private CBBase {
        CallbackParams(T *value, T min, T max, T scale)
        : value(value)
        , min(min)
        , max(max)
        , scale(scale)
        {
            static_assert(std::is_arithmetic<T>::value, "Must be an arithmetic type.");
        }

        ~CallbackParams() override { }

        T *value;
        T min, max, scale;
    };

    template <typename T>
    struct CallbackFunction: private CBBase {
        CallbackFunction(const std::function<void(T)> &func, T min, T max, T scale)
        : func(func)
        , min(min)
        , max(max)
        , scale(scale)
        {
            static_assert(std::is_arithmetic<T>::value, "Must be an arithmetic type.");
        }

        ~CallbackFunction() override { }

        std::function<void(T)> func;
        T min, max, scale;
    };

public:
    Tuner(const std::string &window_name)
    : w_(window_name)
    {
#ifdef HAVE_OPENGL // TODO: replace with CV-specific opengl and get rid of
                   // try-catch
        try {
            cv::namedWindow(w_, cv::WINDOW_OPENGL & cv::WINDOW_KEEPRATIO);
        } catch (cv::Exception &ex) {
            whoWarn(w_,
                    "OpenCV not compiled with OpenGL support. Falling "
                    "back to OpenCV's display driver.\n");
            cv::namedWindow(w_, cv::WINDOW_NORMAL & cv::WINDOW_KEEPRATIO);
        }
#else
        cv::namedWindow(w_, cv::WINDOW_NORMAL);
#endif
    }

    ~Tuner()
    {
        for (auto &p : cb_params_)
            delete static_cast<CBBase *>(p); 
    }

    template <typename T>
    void registerParameter(
        T *param, const char *name, T min, T max, T value = 0, T scale = 1)
    {
        auto cb = [](int value, void *pars) {
            auto p = static_cast<CallbackParams<T> *>(pars);
            auto new_value = static_cast<T>(value) / p->scale;
            if (new_value <= p->max && new_value >= p->min)
                *p->value = new_value;
        };

        auto tmp = static_cast<int>(scale * value);
        cb_params_.push_back(new CallbackParams<T>(param, min, max, scale));
        cv::createTrackbar(name, w_, &tmp, scale * max, cb, cb_params_.back());
    }

    void tune(const oat::Frame &frame,
              const oat::Pose &pose,
              const cv::Matx33d &K,
              const std::vector<double> &D)
    {
        // Make sure this is color image
        oat::Frame col_frame = frame;
        oat::convertColor(frame, col_frame, PIX_BGR);

        // Message to be printed on screen
        std::stringstream msg; 

        if (pose.found) {

            double length;
            if (pose.unit_of_length == Pose::DistanceUnit::Pixels)
                length = 50;
            if (pose.unit_of_length == Pose::DistanceUnit::Meters)
                length = 0.1;

            // Make 3D axis
            std::vector<cv::Point3f> axis_3d;
            axis_3d.push_back(cv::Point3f(0, 0, 0));
            axis_3d.push_back(cv::Point3f(length, 0, 0));
            axis_3d.push_back(cv::Point3f(0, length, 0));
            axis_3d.push_back(cv::Point3f(0, 0, length));
            std::vector<cv::Point2f> frame_axis;
            cv::projectPoints(axis_3d,
                              pose.orientation<cv::Vec3d>(),
                              pose.position<cv::Vec3d>(),
                              K,
                              D,
                              frame_axis);

            // Draw axis
            cv::line(col_frame, frame_axis[0], frame_axis[1], cv::Scalar(0, 0, 255), 3);
            cv::line(col_frame, frame_axis[0], frame_axis[2], cv::Scalar(0, 255, 0), 3);

            // TODO: Find out if Z is degenerate. If so, don't plot.
            if (pose.orientation<std::array<double, 4>>()[2] != 0)
                cv::line(col_frame, frame_axis[0], frame_axis[3], cv::Scalar(255, 0, 0), 3);

            auto p = pose.position<std::array<double, 3>>();
            auto o = pose.orientation<std::array<double, 4>>();
            msg << "[" << p[0] << ", " << p[1] << ", " << p[2] << "], "
                << "[" << o[0] << ", " << o[1] << ", " << o[2] << ", " << o[3] << "]";
        } else {
            msg << "Not found";
        }

        int baseline = 0;
        cv::Size textSize = cv::getTextSize(msg.str(), 1, 1, 1, &baseline);
        cv::Point text_origin(col_frame.cols - textSize.width - 10,
                              col_frame.rows - 2 * baseline - 10);

        cv::putText(col_frame, msg.str(), text_origin, 1, 1, cv::Scalar(0, 255, 255));

        cv::imshow(w_, col_frame);
        cv::waitKey(1);
    }

private:
    std::vector<void *> cb_params_;
    std::string w_;
};

}      /* namespace oat */
#endif /* OAT_TUNER_H */
