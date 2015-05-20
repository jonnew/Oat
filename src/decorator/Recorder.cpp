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

#include <string>

#include "Recorder.h"

Recorder::Recorder(const std::vector<std::string>& save_path, bool append_date) {

    openFile(save_path, append_date);
}

Recorder::writeFrameToVideo(const cv::Mat&);
Recorder::writePositionsToFile(const std::vector<datatypes::Position2D>&);

Recorder::openFile() {
    //full_path = ...	
}
