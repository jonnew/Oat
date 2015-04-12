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

using namespace boost::interprocess;

MatClient::MatClient(const std::string source_name) :
  name(source_name)
, shmem_name(source_name + "_sh_mem")
, shobj_name(source_name + "_sh_obj")
, shared_object_found(false)
{ }

MatClient::MatClient(const MatClient& orig) { }

MatClient::~MatClient() { }

void MatClient::findSharedMat() {

    while (!shared_object_found) {
        try {

            shared_memory = managed_shared_memory(open_only, shmem_name.c_str());
            shared_mat_header = shared_memory.find<shmem::SharedCVMatHeader>(shobj_name.c_str()).first;
            shared_object_found = true;

        } catch (...) {
            std::cout << "Waiting for source \"" + name + "\" to start..." << std::endl;
            usleep(100000);
        } 
    }
    
    // Pass mutex to the scoped sharable_lock. 
    lock = makeLock();
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
cv::Mat MatClient::get_shared_mat() {

    if (!shared_object_found) {
        findSharedMat(); // Creates lock targeting shared_mat_header->mutex, but does not engage
    }
    
    //lock.lock();
    //mat = shared_cvmat->get_value();
    return mat; // User responsible for calling wait after they get, and process this result!
}

void MatClient::wait() {
    
    shared_mat_header->cond_var.wait(lock);
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

