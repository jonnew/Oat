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
//#include "cpptoml.h"

#include "../../lib/shmem/MatClient.h"
#include "../../lib/shmem/MatClient.cpp"

using namespace boost::interprocess;

Viewer::Viewer(std::string source_name) : MatClient(source_name) 
{ 
// TODO: Settings specify window location

}

void Viewer::showImage() {
    
    showImage(cli_name);
}

void Viewer::showImage(const std::string title) {
    
    if (!cli_shared_mat_created) {
        findSharedMat();
    }
    
    sharable_lock<interprocess_sharable_mutex> lock(cli_shared_mat_header->mutex);

    cv::imshow(title, get_shared_mat());
    cv::waitKey(1);
    
    cli_shared_mat_header->cond_var.notify_all();
    cli_shared_mat_header->cond_var.wait(lock);
}