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

#ifndef BACKGROUNDSUBTRACTOR_H
#define	BACKGROUNDSUBTRACTOR_H

#include <string>
#include <opencv2/core/mat.hpp>

#include "../../lib/shmem/MatClient.h"
#include "../../lib/shmem/MatServer.h"

class BackgroundSubtractor {
    
public:
    
    BackgroundSubtractor(const std::string source_name, const std::string sink_name);

    void setBackgroundImage(void);
    void subtractBackground(void);
    
    // Detectors must be interruptable
    void stop(void) { frame_sink.set_running(false); }

private:

    // The background image used for subtraction
    bool background_set = false;
    cv::Mat current_frame;
    cv::Mat current_raw_frame;
    cv::Mat background_img;
    
    // Mat client object for receiving frames
    MatClient frame_source;
    
    // Mat server for sending processed frames
    MatServer frame_sink;

};

#endif	/* BACKGROUNDSUBTRACTOR_H */

