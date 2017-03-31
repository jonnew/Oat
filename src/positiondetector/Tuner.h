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

#include <functional>
#include <string>
#include <vector>
#include <type_traits>

#include "../../lib/datatypes/Frame.h"
#include "../../lib/datatypes/Pose.h"

#define TUNE tuner_->registerParameter

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

    explicit Tuner(const std::string &window_name);
    ~Tuner();

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

    void tune(const cv::Mat &mat,
              const oat::Pose &pose,
              const cv::Matx33d &K = cv::Matx33d::eye(),
              const std::vector<double> &D = {0, 0, 0, 0, 0, 0, 0, 0});

private:
    std::vector<void *> cb_params_;
    std::string w_;
};

}      /* namespace oat */
#endif /* OAT_TUNER_H */
