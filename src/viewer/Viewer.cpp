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

#include "Viewer.h"

#include <string>
#include <boost/interprocess/sync/sharable_lock.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../../lib/shmem/MatClient.h"
#include "../../lib/shmem/MatClient.cpp"

using namespace boost::interprocess;

Viewer::Viewer(std::string source_name) : frame_source(source_name) { }

void Viewer::showImage() {
    
    showImage(frame_source.get_name());
}

void Viewer::showImage(const std::string title) {
    
    cv::imshow(title, frame_source.get_shared_mat());
    cv::waitKey(1);
    
    // Wait for signal that next frame is ready
    frame_source.wait();
}