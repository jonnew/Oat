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

#ifndef VIEWER_H
#define VIEWER_H

#include <chrono>
#include <string>
#include <opencv2/core/mat.hpp>

#include "../../lib/shmem/Signals.h"
#include "../../lib/shmem/SharedCVMatHeader.h"
#include "../../lib/shmem/MatClient.h"

class Viewer {
    
    typedef std::chrono::high_resolution_clock Clock;
    typedef std::chrono::milliseconds milliseconds;
    
public:
    Viewer(const std::string& frame_source_name, 
           std::string& save_path,
           const std::string& file_name);

    oat::ServerRunState showImage(void);
    oat::ServerRunState showImage(std::string title);
    void stop(void);

    // Accessors
    std::string get_name(void) { return name; }
    
private:

    // Viewer name
    std::string name;

    // Image data
    cv::Mat current_frame;;

    // Mat client object for receiving frames
    oat::MatClient frame_source;
    
    // minimum viewer refresh period
    Clock::time_point tick, tock;
    const milliseconds min_update_period;
    
    // Used to request a snapshot of the current image, saved to disk
    std::string frame_fid;
    std::string save_path;
    std::string file_name;
    bool append_date;
    std::vector<int> compression_params;
    
    std::string makeFileName(void);
};

#endif //VIEWER_H
