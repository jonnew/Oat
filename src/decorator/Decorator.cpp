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
#include <ctime>

Decorator::Decorator(const std::vector<std::string>& position_source_names,
        const std::string& frame_source_name,
        const std::string& frame_sink_name) :
  name("decorator[" + frame_source_name + "->" + frame_sink_name + "]")
, frame_source(frame_source_name)
, frame_read_success(false)
, client_idx(0)
, frame_sink(frame_sink_name)
, decorate_position(true)
, print_timestamp(false)
, print_sample_number(false)
, encode_sample_number(false)
, font_color(0, 255, 0) {

    if (!position_source_names.empty()) {
        for (auto &source_name : position_source_names) {

            position_sources.push_back(new oat::SMClient<oat::Position2D>(source_name));
            source_positions.push_back(new oat::Position2D);
        }
    } else {
        decorate_position = false;
    }
}

Decorator::~Decorator() {

    // Release resources
    for (auto &position_source : position_sources) {
        delete position_source;
    }
}

bool Decorator::decorateFrame() {

    // Are all sources running?
    bool sources_running = true;
    
    // Get the image to be decorated
    if (!frame_read_success) {
        frame_read_success = frame_source.getSharedMat(current_frame);
        sources_running &= (frame_source.getSourceRunState() == oat::ServerRunState::RUNNING);
    }
        

    // Get the current positions
    while (client_idx < position_sources.size() && decorate_position) {

        if (!(position_sources[client_idx]->getSharedObject(*source_positions[client_idx]))) {
            return sources_running;
        }

        sources_running &= (position_sources[client_idx]->getSourceRunState() == oat::ServerRunState::RUNNING);
        client_idx++;
    }

    if (!frame_read_success) {
        return sources_running;
    }
    
    // Reset the position client read counter
    client_idx = 0;
    frame_read_success = false;

    // Decorated image
    drawSymbols();

    // Serve the finished product
    frame_sink.pushMat(current_frame, frame_source.get_current_sample_number());
    
    return sources_running;
}

void Decorator::drawSymbols() {

    if (decorate_position) {
        drawPosition();
        drawHeadDirection();
        drawVelocity();
    }

    if (print_timestamp) {
        printTimeStamp();
    }

    if (print_sample_number) {
        printSampleNumber();
    }

    if (encode_sample_number) {
        encodeSampleNumber();
    }
}

// TODO: project 3rd dimension

void Decorator::drawPosition() {

    for (auto position : source_positions) {
        if (position->position_valid) {
            cv::circle(current_frame, position->position, position_circle_radius, cv::Scalar(0, 0, 255), 2);
        }
    }
}

// TODO: project 3rd dimension

void Decorator::drawHeadDirection() {
    for (auto position : source_positions) {
        if (position->position_valid && position->heading_valid) {
            cv::Point2d start = position->position - (head_dir_line_length * position->heading);
            cv::Point2d end = position->position + (head_dir_line_length * position->heading);
            cv::line(current_frame, start, end, cv::Scalar(255, 255, 255), 2, 8);
        }
    }
}

// TODO: project 3rd dimension

void Decorator::drawVelocity() {
    for (auto position : source_positions) {
        if (position->velocity_valid && position->position_valid) {
            cv::Point2d end = position->position + (velocity_scale_factor * position->velocity);
            cv::line(current_frame, position->position, end, cv::Scalar(0, 255, 0), 2, 8);
        }
    }
}

void Decorator::printTimeStamp() {

    std::time_t raw_time;
    struct tm * time_info;
    char buffer[80];

    std::time(&raw_time);
    time_info = std::localtime(&raw_time);

    std::strftime(buffer, 80, "%c", time_info);

    cv::Point text_origin(current_frame.cols - 230, current_frame.rows - 10);
    cv::putText(current_frame, std::string(buffer), text_origin, 1, font_scale, font_color);
}

void Decorator::printSampleNumber() {

    cv::Point text_origin(10, current_frame.rows - 10);
    cv::putText(current_frame, std::to_string(frame_source.get_current_sample_number()), text_origin, 1, font_scale, font_color);
}

/**
 * Encode the current sample number into the first row of the matrix
 */
void Decorator::encodeSampleNumber() {

    uint32_t sample_number = frame_source.get_current_sample_number();

    int column = 0;
    for (int shift = 0; shift < 32; shift++) {

        cv::Mat sub_square = current_frame.colRange(column, column + encode_bit_size).rowRange(0, encode_bit_size);

        if (sample_number & 0x1) {

            cv::Mat true_mat(encode_bit_size, encode_bit_size, current_frame.type(), CV_RGB(255, 255, 255));
            true_mat.copyTo(sub_square);

        } else {

            cv::Mat false_mat = cv::Mat::zeros(encode_bit_size, encode_bit_size, current_frame.type());
            false_mat.copyTo(sub_square);
        }

        sample_number >>= 1;
        column += encode_bit_size;
    }
}
