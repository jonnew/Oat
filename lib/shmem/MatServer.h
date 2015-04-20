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
#include <thread>
#include <mutex>
#include <condition_variable>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <opencv2/core/mat.hpp>

#include "SharedCVMatHeader.h"

class MatServer {
    
public:
    MatServer(const std::string sink_name);
    MatServer(const MatServer& orig);
    virtual ~MatServer();
    
    void createSharedMat(const cv::Mat& model); // TODO: encapsulate in the SharedMatHeader object
    void pushMat(const cv::Mat& mat);
    
    // Accessors  // TODO: Assess whether you really need these and get rid of them if not. 
    bool is_running(void) { return running; };
    void set_running(bool value) { running = value; } 
    std::string get_name(void) { return name; }
    
private:
    
    const std::string start_mutex_name, start_condition_name;
    std::string name;
    boost::lockfree::spsc_queue<cv::Mat, boost::lockfree::capacity<1024> > mat_buffer;
    std::thread server_thread;
    std::mutex server_mutex;
    std::condition_variable serve_condition;
    bool running; // Server running
    shmem::SharedCVMatHeader* shared_mat_header;
    bool shared_object_created;
    void* shared_mat_data_ptr;
    int data_size; // Size of raw mat data in bytes
    
    const std::string shmem_name, shobj_name;
    boost::interprocess::managed_shared_memory shared_memory; // TODO:  encapsulate in the SharedMatHeader object
    
    /**
     * Synchronized shared memory service.
     * @param mat
     */
    void serveMatFromBuffer(void);

};

#endif	/* MATSERVER_H */

