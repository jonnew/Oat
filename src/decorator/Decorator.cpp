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

Decorator::Decorator(std::string position_source_name,
        std::string frame_source_name,
        std::string frame_sink_name) :
  frame_source(frame_source_name)
, position_source(position_source_name)
, frame_sink(frame_sink_name)
, current_processing_stage(0) {

    frame_source.findSharedMat();
    position_source.findSharedObject();
}

void Decorator::decorateAndServeImage() {

    // Get the current image
    switch (current_processing_stage) {
        case 0:

            if (!frame_source.getSharedMat(image)) {
                return;
            }

            // Fall through
            current_processing_stage = 1;

        case 1:
            // Get the current position
            if (!position_source.getSharedObject(position)) {
                return;
            }

            // Fall through
            current_processing_stage = 0;
    }

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
    if (position.position_valid) {
        cv::circle(image, position.position, position_circle_radius, cv::Scalar(0, 0, 255), 2);
    }
}

// TODO: project 3rd dimension

void Decorator::drawHeadDirection() {
    if (position.position_valid && position.head_direction_valid) {
        cv::Point2d start = position.position - (head_dir_line_length * position.head_direction);
        cv::Point2d end = position.position + (head_dir_line_length * position.head_direction);
        cv::line(image, start, end, cv::Scalar(255, 255, 255), 2, 8);
    }
}

// TODO: project 3rd dimension

void Decorator::drawVelocity() {
    if (position.velocity_valid && position.position_valid) {
        cv::Point2d end = position.position + (velocity_scale_factor * position.velocity);
        cv::line(image, position.position, end, cv::Scalar(0, 255, 0), 2, 8);
    }
}