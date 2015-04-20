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

#include "MatServer.h"

//#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
//#include <boost/interprocess/sync/named_condition_any.hpp>
//#include <boost/interprocess/sync/named_sharable_mutex.hpp>
#include <boost/interprocess/sync/sharable_lock.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
//#include <boost/interprocess/sync/named_condition.hpp>
//#include <boost/interprocess/sync/named_mutex.hpp>

#include "SharedCVMatHeader.h"
#include "SharedCVMatHeader.cpp" // TODO: Why???

using namespace boost::interprocess;

MatServer::MatServer(const std::string sink_name) :
  name(sink_name)
, shmem_name(sink_name + "_sh_mem")
, shobj_name(sink_name + "_sh_obj")
, start_mutex_name(sink_name + "_start_mtx")
, start_condition_name(sink_name + "_start_cv")
, shared_object_created(false)
, running(true) {

    // Start the server thread
    server_thread = std::thread(&MatServer::serveMatFromBuffer, this);
}

MatServer::MatServer(const MatServer& orig) { }

MatServer::~MatServer() {

    // Join the server thread back with the main one
    server_thread.join();
		
    // Remove_shared_memory on object destruction
    shared_mat_header->new_data_condition.notify_all();
    shared_memory_object::remove(shmem_name.c_str());
}

void MatServer::createSharedMat(const cv::Mat& model) {
    
    // Clean up any potential leftovers
    shared_memory_object::remove(shmem_name.c_str());

    data_size = model.total() * model.elemSize();
    try {

        // Define shared memory
        shared_memory = managed_shared_memory(open_or_create,
                shmem_name.c_str(),
                data_size + sizeof (shmem::SharedCVMatHeader) + 1024);

        // Make the shared object
        shared_mat_header = shared_memory.find_or_construct<shmem::SharedCVMatHeader>(shobj_name.c_str())();

    } catch (bad_alloc &ex) {
        std::cerr << ex.what() << '\n';
        exit(EXIT_FAILURE); // TODO: exit does not unwind the stack to take care of destructing shared memory objects
    }
    
    shared_mat_header->buildHeader(shared_memory, model);
    
    shared_object_created = true;
    
    // TODO: I cannot get this to work, even though it would be far better than 
    // my !shared_mat_header->server_ready hack ...
//    named_mutex start_mutex(open_or_create, start_mutex_name.c_str());
//    named_condition start_condition(open_or_create, start_condition_name.c_str());
//    scoped_lock<named_mutex> start_lock(start_mutex);
//    start_condition.notify_all();
//    boost::system_time timeout = boost::get_system_time() + boost::posix_time::milliseconds(1);
//    start_condition.timed_wait(start_lock, timeout);
    
}

void MatServer::pushMat(const cv::Mat& mat) {
    
    // Push data onto ring buffer
    mat_buffer.push(mat.clone());
    
    // notify server thread that data is available
    serve_condition.notify_one();
}

void MatServer::serveMatFromBuffer() {

    while (running) {
        
        // Proceed only if mat_buffer has data
        std::unique_lock<std::mutex> lk(server_mutex);
        serve_condition.wait(lk); 

        // Here we must attempt to clear the whole buffer before waiting again.
        cv::Mat mat;
        while (mat_buffer.pop(mat)) {

            // Create shared mat object if not done already
            if (!shared_object_created) {
                createSharedMat(mat);
            }

            // Exclusive scoped_lock on the shared_mat_header->mutex
            scoped_lock<interprocess_sharable_mutex> lock(shared_mat_header->mutex);

            // Perform write in shared memory 
            shared_mat_header->set_value(mat);

            // Notify all client processes they can now access the data
            shared_mat_header->new_data_condition.notify_all();

            // Wait for clients to finish with data, will only occur when this thread
            // can regain an exclusive lock on the mutex
            shared_mat_header->new_data_condition.wait(lock);
            
        } // Scoped lock is released
    }
    
    
} // Lock is released on scope exit

//void MatServer::set_name(const std::string sink_name) {
//    
//    // Make sure we are not already attached to some source
//    if (!shared_object_created) {
//        name = sink_name;
//        shmem_name = sink_name + "_sh_mem";
//        shobj_name = sink_name + "_sh_obj";
//    } else {
//        std::cerr << "Cannot edit the sink name because we are already reading from \"" + name + "\".";
//    }
//}
