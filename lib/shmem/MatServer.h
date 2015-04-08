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

#ifndef MATSERVER_H
#define	MATSERVER_H

#include <string>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <opencv2/core/mat.hpp>

#include "SharedMat.h"

namespace ip = boost::interprocess;

class MatServer {
    
public:
    MatServer(const std::string sink_name);
    MatServer(const MatServer& orig);
    virtual ~MatServer();
    
    void createSharedMat(cv::Mat model);
    
    void notifyAll(void);
    void wait(void);
    void notifyAllAndWait(void);
    
    // Accessors
    void set_shared_mat(cv::Mat mat);
    bool is_shared_mat_created(void) { return shared_mat_created; }
    std::string get_name(void) { return name; }
    
private:
    
    ip::scoped_lock<ip::interprocess_sharable_mutex> makeLock();
    
    std::string name;
    shmem::SharedMatHeader* shared_mat_header;
    bool shared_mat_created = false;
    void* shared_mat_data_ptr;
    int data_size; // Size of raw mat data in bytes
    
    std::string shmem_name, shobj_name;
    ip::managed_shared_memory shared_memory;
    ip::scoped_lock<ip::interprocess_sharable_mutex> lock; 

};

#endif	/* MATSERVER_H */

