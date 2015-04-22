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

#ifndef MATCLIENT_H
#define	MATCLIENT_H

#include <string>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/sharable_lock.hpp>
#include <boost/interprocess/sync/interprocess_sharable_mutex.hpp>
#include <opencv2/core/mat.hpp>


#include "SharedCVMatHeader.h"

class MatClient {
public:
    //MatClient(void);
    MatClient(const std::string server_name);
    MatClient(const MatClient& orig);
    virtual ~MatClient();
    
    // Find cv::Mat object in shared memory
    int findSharedMat(void);
    
    // get cv::Mat out of shared memory
    void getSharedMat(cv::Mat& value);
    
    // Auto notification to exit wait()
    void notifySelf(void);
     
    // Accessors
    std::string get_name(void) { return name; }
    
private:
    
    std::string name;
    shmem::SharedCVMatHeader* shared_mat_header;
    bool shared_object_found, mat_attached_to_header;
    int data_size; // Size of raw mat data in bytes

    // Shared mat object, constructed from the shared_mat_header
    cv::Mat mat;

    const std::string shmem_name, shobj_name;
    boost::interprocess::managed_shared_memory shared_memory;
};

#endif	/* MATCLIENT_H */

