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
#include <cmath>

Decorator::Decorator(const std::vector<std::string>& position_source_names,
        const std::string& frame_source_name,
        const std::string& frame_sink_name) :
  name("decorator[" + frame_source_name + "->" + frame_sink_name + "]")
, frame_source(frame_source_name)
, frame_read_success(false)
, frame_sink(frame_sink_name)
, number_of_position_sources(position_source_names.size())
, position_read_required(number_of_position_sources)
, sources_eof(false)
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
    
    position_read_required.set();
}

Decorator::~Decorator() {

    // Release resources
    for (auto &value : position_sources) {
        delete value;
    }
    
    for (auto &value : source_positions) {
        delete value;
    }
}

bool Decorator::decorateFrame() {

    // Make sure all sources are still running
    sources_eof |= (frame_source.getSourceRunState() 
        == oat::ServerRunState::END);

    for (int i = 0; i < number_of_position_sources; i++) {

        sources_eof |= (position_sources[i]->getSourceRunState()
                == oat::ServerRunState::END);
    }
    
    // Get the image to be decorated
    if (!frame_read_success) {
        frame_read_success = frame_source.getSharedMat(current_frame);
    }
    
    boost::dynamic_bitset<>::size_type i = position_read_required.find_first();

    // Get current positions
    while (i < number_of_position_sources) {

        position_read_required[i] =
                !position_sources[i]->getSharedObject(*source_positions[i]);
        
        i = position_read_required.find_next(i);
        
    }

    // If we have not finished reading _any_ of the clients, we cannot proceed
    if (frame_read_success && position_read_required.none()) {

        // Reset the frame and position client read counter
        frame_read_success = false;
        position_read_required.set();

        // Decorated image
        drawSymbols();

        // Serve the finished product
        frame_sink.pushMat(current_frame, frame_source.get_current_sample_number());
    }
    
    return sources_eof;
}

void Decorator::drawSymbols() {

    if (decorate_position) {
        drawPosition();
        drawHeading();
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

void Decorator::drawPosition() {

    for (auto position : source_positions) {
        if (position->position_valid) {
            cv::circle(current_frame, position->position, position_circle_radius, cv::Scalar(0, 0, 255), 2);
        }
    }
}

void Decorator::drawHeading() {
    for (auto position : source_positions) {
        if (position->position_valid && position->heading_valid) {
            cv::Point2d start = position->position - (heading_line_length * position->heading);
            cv::Point2d end = position->position + (heading_line_length * position->heading);

            // Draw arrow
            cv::line(current_frame, start, end, cv::Scalar(255, 0, 0), 2, 8);
            double angle = std::atan2((double) start.y - end.y, (double) start.x - end.x);
            start.x = end.x + heading_arrow_length * std::cos(angle + PI / 4);
            start.y = end.y + heading_arrow_length * std::sin(angle + PI / 4);
            cv::line(current_frame, start, end, cv::Scalar(255, 0, 0), 2, 8);
            start.x = end.x + heading_arrow_length * std::cos(angle - PI / 4);
            start.y = end.y + heading_arrow_length * std::sin(angle - PI / 4);
            cv::line(current_frame, start, end, cv::Scalar(255, 0, 0), 2, 8);
        }
    }
}

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
    cv::putText(current_frame, 
                std::to_string(frame_source.get_current_sample_number()), 
                text_origin, 
                1, 
                font_scale, 
                font_color);
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
