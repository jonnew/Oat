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

#include "MatClient.h"

#include <unistd.h>
#include <boost/thread.hpp>

using namespace boost::interprocess;

MatClient::MatClient(const std::string source_name) :
  name(source_name)
, shmem_name(source_name + "_sh_mem")
, shobj_name(source_name + "_sh_obj")
, shared_object_found(false)
, terminated(false) { }

MatClient::MatClient(const MatClient& orig) { }

MatClient::~MatClient() { }

void MatClient::findSharedMat() {

    while (!shared_object_found) {
        try {

            shared_memory = managed_shared_memory(open_only, shmem_name.c_str());
            shared_mat_header = shared_memory.find<shmem::SharedCVMatHeader>(shobj_name.c_str()).first;
            shared_object_found = true;

        } catch (interprocess_exception& e) {
            usleep(100000); // Wait for shared memory to be created by SOURCE
            //std::cerr << "Error: " << e.what() << "\n";
            //std::cerr << "  This is likely due to the SOURCE, \"" << name << "\", not being started.\n";
            //std::cerr << "  Did you start the SOURCE, \"" << name << "\", before staring this client?" << std::endl;
            //exit(EXIT_FAILURE); // TODO: exit does not unwind the stack to take care of destructing shared memory objects
        } 
        
        if (terminated)
            exit(EXIT_FAILURE); // Nothing to clean, so we are OK to exit.
    }

    // Pass mutex to the scoped sharable_lock. 
    lock = makeLock(); // This will block until the lock has sharable access to the mutex
    shared_mat_header->attachMatToHeader(shared_memory, mat);
}

/**
 * Engages shared lock on the unnamed shared mutex residing in the shared_mat_header
 * and returns the shared cv::Mat object. The user is responsible for calling
 * wait() following this function call, and after they are finished using the 
 * cv::Mat object to (1) release the shared lock on the mutex and (2) to puase this
 * thread until the shared cv::Mat object is updated by the server.
 * 
 * @return shared cv::Mat. Do not write on this object.
 */
cv::Mat MatClient::get_value() {

    if (!shared_object_found) {
        findSharedMat(); // Creates lock targeting shared_mat_header->mutex, and engages
    }
    
    // Wait for notification from SOURCE to grant access to shared memory
    shared_mat_header->new_data_condition.wait(lock); 

    return mat; // User responsible for calling wait after they get, and process this result!
}

sharable_lock<interprocess_sharable_mutex> MatClient::makeLock(void) {
    sharable_lock<interprocess_sharable_mutex> sl(shared_mat_header->mutex); // defer_lock
    return sl;
}

void MatClient::set_source(const std::string source_name) {

    // Make sure we are not already attached to some source
    if (!shared_object_found) {
        name = source_name;
        shmem_name = source_name + "_sh_mem";
        shobj_name = source_name + "_sh_obj";
    } else {
        std::cerr << "Cannot edit the source name because we are already reading from \"" + name + "\".";
    }
}

void MatClient::notifySelf(){
    
    if (shared_object_found) {
        shared_mat_header->new_data_condition.notify_one();
    }
    terminated = true;
}

