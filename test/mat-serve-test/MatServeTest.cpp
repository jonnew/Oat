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

#include "MatServeTest.h"

#include <string>
#include <opencv2/core/core.hpp>
//#include "cpptoml.h"

//#include "MatServer.h"
#include "../../lib/shmem/SharedMat.h"
#include "../../lib/shmem/MatServer.h"
#include "../../lib/shmem/MatServer.cpp"

MatServeTest::MatServeTest(std::string server_name) : MatServer(server_name) { }

int MatServeTest::openVideo(const std::string fid) {

    cap.open(fid); // open the default video
    if (!cap.isOpened()) // check if we succeeded
        return -1;
}

int MatServeTest::serveMat() {

    cap >> mat; // get a new frame from video
    if (mat.empty()) {
        return 0;
        usleep(100000);
    }
    
    // Thread-safe set to shared mat object
    set_shared_mat(mat);
    
    return 1;

}