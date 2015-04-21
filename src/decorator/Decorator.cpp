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
, frame_sink(frame_sink_name) {
    
    frame_source.findSharedMat();
    position_source.findSharedObject();
}

void Decorator::decorateImage() {

    // Get the current position
    position = position_source.get_value();

    // Get the current image
    image = frame_source.get_value().clone();

    // Decorate
    drawSymbols();

}

void Decorator::serveImage() {
    frame_sink.pushMat(image);
}

void Decorator::stop() {
    frame_source.notifySelf();
    position_source.notifySelf();
}

void Decorator::drawSymbols() {

    drawPosition();
    drawHeadDirection();
    drawVelocity();
}

// TODO: project 3rd dimension
void Decorator::drawPosition() {
    if (position.position_valid) {
        cv::Point2f pos(position.position.x, position.position.y);
        cv::circle(image, pos, position_circle_radius, cv::Scalar(0,0,255), 2);
    }
}

// TODO: project 3rd dimension
void Decorator::drawHeadDirection() {
    if (position.position_valid && position.head_direction_valid) {
        cv::Point3f start = position.position - (head_dir_line_length * position.head_direction);
        cv::Point3f end = position.position + (head_dir_line_length * position.head_direction);
        cv::line(image, cv::Point2f(start.x,start.y), cv::Point2f(end.x,end.y), cv::Scalar(255, 255, 255), 2, 8);
    }
}

// TODO: project 3rd dimension
void Decorator::drawVelocity() {
    if (position.velocity_valid && position.position_valid) {
        cv::Point3f end = position.position + (velocity_scale_factor * position.velocity);
        cv::Point2f pos(position.position.x, position.position.y);
        cv::line(image, pos, cv::Point2f(end.x,end.y), cv::Scalar(0, 255, 0), 2, 8);
    }
}