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

#ifndef MATSERVETEST_H
#define MATSERVETEST_H

#include <string>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

#include "../../lib/shmem/MatServer.h"

class MatServeTest {
   
public:
    MatServeTest(std::string server_name);

    int openVideo(const std::string fid);
    int serveMat(void);
   
private:
    
    int data_size;
    void* shared_mat_data_ptr;
    cv::VideoCapture cap;
    
    // Image data
    cv::Mat mat;
    
    // Shmem server
    uint32_t sample;
    shmem::MatServer mat_source;

};

#endif //MATSERVETEST_H
