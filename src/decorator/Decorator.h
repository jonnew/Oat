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

class Decorator {
   
public:
    Decorator(std::string position_source_name, 
              std::string frame_source_name,             
              std::string frame_sink_name);
    
    void decorateAndServeImage(void);
    
    void stop(void) {frame_sink.set_running(false); }
   
private:
    
    // Image data
    cv::Mat image;
    
    // For multi-server processing, we need to keep track of all the servers
    // we have finished reading from each processing step
    int current_processing_stage;
    
    // Current position
    shmem::Position position;
    
    // Mat client object for receiving frames
    MatClient frame_source;
    
    // Position client for getting current position info
    shmem::SMClient<shmem::Position> position_source;
    
    // Mat server for sending decorated frames
    MatServer frame_sink;
    
    // Drawing constants // TODO: These will need to change if image information
    // starts coming in world-value units
    const float position_circle_radius = 5.0;
    const float head_dir_line_length = 25.0;
    const float velocity_scale_factor = 10.0;
    
    void drawPosition();
    void drawHeadDirection();
    void drawVelocity();
    void drawSymbols();
};

#endif //VIEWER_H
