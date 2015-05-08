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

#include <string>
#include <opencv2/core/mat.hpp>

#include "../../lib/shmem/SharedCVMatHeader.h"
#include "../../lib/shmem/MatClient.h"

class Viewer {
public:
    Viewer(std::string server_name);

    void showImage(void);
    void showImage(std::string title);
    void stop(void);

    // Accessors

    std::string get_name(void) {
        return name;
    }
    
private:

    // Viewer name
    std::string name;

    // Image data
    cv::Mat current_frame;;

    // Mat client object for receiving frames
    MatClient frame_source;
};

#endif //VIEWER_H
