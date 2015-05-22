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

#include "../../lib/shmem/SMClient.h"
#include "../../lib/shmem/MatClient.h"
#include "../../lib/shmem/MatServer.h"
#include "../../lib/datatypes/Position2D.h"

class Decorator { // TODO: Position2D -> Position somehow
    
public:
    
    Decorator(const std::vector<std::string>& position_source_names,
              const std::string& frame_source_name,
              const std::string& frame_sink_name);

    ~Decorator(void);
    
    void decorateAndServeImage(void);
    void stop(void) { frame_sink.set_running(false); }
    
    //Accessors
    std::string get_name(void) const { return name; }

private:

    std::string name;

    // Image data
    cv::Mat current_frame;

    // Mat client object for receiving frames
    MatClient frame_source;
    cv::Size frame_size;
    bool have_current_frame;

    // For multi-source processing, we need to keep track of all the sources
    // we have finished reading from each processing step
    std::vector<shmem::SMClient<datatypes::Position2D> >::size_type client_idx;

    // Positions to be added to the image stream
    std::vector<datatypes::Position2D* > source_positions;
    std::vector<shmem::SMClient<datatypes::Position2D>* > position_sources;

    // Mat server for sending decorated frames
    MatServer frame_sink;

    // Drawing constants 
    // TODO: These may need to become a bit more sophisticated or user defined
    bool decorate_position;
    const float position_circle_radius = 5.0;
    const float head_dir_line_length = 25.0;
    const float velocity_scale_factor = 0.1;
    const double font_scale = 1.0; 
    const cv::Scalar font_color;
    const int encode_bit_size = 5;

    void drawPosition(void);
    void drawHeadDirection(void);
    void drawVelocity(void);
    void drawSymbols(void);
    void printTimeStamp(void); // TODO: configure position of timestamp text
    void printSampleNumber(void);  // TODO: configure position of sample number text
    void encodeSampleNumber(void);
};

#endif //VIEWER_H
