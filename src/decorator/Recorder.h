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

#ifndef RECORDER_H
#define RECORDER_H

#include <string>

#include "../../lib/datatypes/Position2D.h"

class Recorder {     
public:

    Recorder(const std::vector<std::string>& save_path, bool append_date);

    void writeFrameToVideo(const cv::Mat&);
    void writePositionsToFile(const std::vector<datatypes::Position2D>&);

private:

    std::string full_path;

    void openFile(const std::vector<std::string>& save_path, bool append_date);
};

// RECORDER_H
