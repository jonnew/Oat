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

#include <atomic>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <opencv2/core/mat.hpp>

#include "SharedCVMatHeader.h"

// TODO: Find a why to integrate this with the must much general purpose SMServer
class MatServer {
    
public:
    MatServer(const std::string& sink_name);
    MatServer(const MatServer& orig);
    virtual ~MatServer();
    
    void createSharedMat(const cv::Mat& model); 
    void pushMat(const cv::Mat& mat);

    // Accessors  // TODO: Assess whether you really need these and get rid of them if not. 
    bool is_running(void) { return running; };
    void set_running(bool value) { running = value; } 
    std::string get_name(void) { return name; }

    void set_homography(const cv::Matx33f& value) {
        homography_valid = true;
        homography = value;
    }
    
    bool is_shared_object_created(void) { return shared_object_created; }
    
private:
    
    // Name of this server
    std::string name;
    
    // Buffer
    static const int MATSERVER_BUFFER_SIZE = 100;
    boost::lockfree::spsc_queue
      <cv::Mat, boost::lockfree::capacity<MATSERVER_BUFFER_SIZE> > mat_buffer;
    boost::lockfree::spsc_queue
      <unsigned int, boost::lockfree::capacity<MATSERVER_BUFFER_SIZE> > tick_buffer;
    
    // Timestamp
    unsigned int current_sample;
    unsigned int write_index;
    
    // Server threading
    std::thread server_thread;
    std::mutex server_mutex;
    std::condition_variable serve_condition;
    std::atomic<bool> running; // Server running, can be accessed from multiple threads
    shmem::SharedCVMatHeader* shared_mat_header;
    bool shared_object_created;

    int data_size; // Size of raw mat data in bytes
    
    const std::string shmem_name, shobj_name;
    boost::interprocess::managed_shared_memory shared_memory; 
    
    // Homography from pixels->world
    bool homography_valid;
    cv::Matx33d homography;
    
    /**
     * Synchronized shared memory publication.
     * @param mat
     */
    void serveMatFromBuffer(void);
    
    /**
     * Auto post to this servers own semaphore.wait() to allow threads to unblock
     * in order to ensure proper object destruction.
     * @param mat
     */
    void notifySelf(void);

};

#endif	/* MATSERVER_H */

