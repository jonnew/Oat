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

#ifndef DECORATOR_H
#define DECORATOR_H

#include <string>
#include <boost/dynamic_bitset.hpp>

#include "../../lib/shmem/SMClient.h"
#include "../../lib/shmem/MatClient.h"
#include "../../lib/shmem/MatServer.h"
#include "../../lib/datatypes/Position2D.h"

class Decorator {
    
public:
    
    Decorator(const std::vector<std::string>& position_source_names,
              const std::string& frame_source_name,
              const std::string& frame_sink_name);

    ~Decorator(void);
    
    bool decorateFrame(void);
    
    //Accessors
    void set_print_timestamp(bool value) { print_timestamp = value; }
    void set_print_sample_number(bool value) { print_sample_number = value; }
    void set_encode_sample_number(bool value) { encode_sample_number = value; }
    std::string get_name(void) const { return name; }

private:

    std::string name;

    // Image data
    cv::Mat current_frame;

    // Mat client object for receiving frames
    oat::MatClient frame_source;
    cv::Size frame_size;
    bool frame_read_success;
    
    // Mat server for sending decorated frames
    oat::MatServer frame_sink;
    
    // Positions to be added to the image stream
    std::vector<oat::Position2D* > source_positions;
    std::vector<oat::SMClient<oat::Position2D>* > position_sources;
    boost::dynamic_bitset<>::size_type number_of_position_sources;
    boost::dynamic_bitset<> position_read_required;

    bool sources_eof;

    // Drawing constants 
    // TODO: These may need to become a bit more sophisticated or user defined
    bool decorate_position;
    bool print_timestamp;
    bool print_sample_number;
    bool encode_sample_number;
    const float position_circle_radius = 5.0;
    const float heading_line_length = 20.0;
    const float heading_arrow_length = 5;
    const float velocity_scale_factor = 0.1;
    const double font_scale = 1.0; 
    const cv::Scalar font_color;
    const int encode_bit_size = 5;
    const double PI  = 3.141592653589793238463;

    void drawPosition(void);
    void drawHeading(void);
    void drawVelocity(void);
    void drawSymbols(void);
    void printTimeStamp(void); 
    void printSampleNumber(void);  
    void encodeSampleNumber(void);
};

#endif //VIEWER_H
