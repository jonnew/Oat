//******************************************************************************
//* Copyright (c) Jon Newman (jpnewman at mit snail edu) 
//* All right reserved.
//* This file is part of the Simple Tracker project.
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

#include "Decorator.h"

#include <string>
#include <boost/interprocess/sync/sharable_lock.hpp>
#include <opencv2/opencv.hpp>

Decorator::Decorator(const std::vector<std::string>& position_source_names,
        const std::string& frame_source_name,
        const std::string& frame_sink_name) :
name(frame_sink_name)
, frame_source(frame_source_name)
, frame_sink(frame_sink_name)
, client_idx(0)
, have_current_frame(false) {

    for (auto &source_name : position_source_names) {

        position_sources.push_back(new shmem::SMClient<datatypes::Position2D>(source_name));
        source_positions.push_back(new datatypes::Position2D);
    }
}

Decorator::~Decorator() {

    // Release resources
    for (auto &position_source : position_sources) {
        delete position_source;
    }
}

void Decorator::decorateAndServeImage() {

    // Get the image to be decorated
    // TODO: for some reason, this must come before reading the current positions.
    // If it comes, after, a deadlock will result. I don't know why, which is not
    // good.
    if (!frame_source.getSharedMat(image) && !have_current_frame) {
        return;
    }
    have_current_frame = true;


    // Get the current positions
    while (client_idx < position_sources.size()) {

        if (!(position_sources[client_idx]->getSharedObject(*source_positions[client_idx]))) {
            return;
        }

        client_idx++;
    }

    // Reset the position client read counter
    client_idx = 0;
    have_current_frame = false;

    // Decorated image
    drawSymbols();

    // Serve the finished product
    frame_sink.pushMat(image);
}

void Decorator::drawSymbols() {

    drawPosition();
    drawHeadDirection();
    drawVelocity();
}

// TODO: project 3rd dimension

void Decorator::drawPosition() {

    for (auto position : source_positions) {
        if (position->position_valid) {
            cv::circle(image, position->position, position_circle_radius, cv::Scalar(0, 0, 255), 2);
        }
    }
}

// TODO: project 3rd dimension

void Decorator::drawHeadDirection() {
    for (auto position : source_positions) {
        if (position->position_valid && position->head_direction_valid) {
            cv::Point2d start = position->position - (head_dir_line_length * position->head_direction);
            cv::Point2d end = position->position + (head_dir_line_length * position->head_direction);
            cv::line(image, start, end, cv::Scalar(255, 255, 255), 2, 8);
        }
    }
}

// TODO: project 3rd dimension

void Decorator::drawVelocity() {
    for (auto position : source_positions) {
        if (position->velocity_valid && position->position_valid) {
            cv::Point2d end = position->position + (velocity_scale_factor * position->velocity);
            cv::line(image, position->position, end, cv::Scalar(0, 255, 0), 2, 8);
        }
    }
}