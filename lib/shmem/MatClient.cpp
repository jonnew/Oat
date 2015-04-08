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
, shared_mat_created(false)
{ }

MatClient::MatClient(const MatClient& orig) { }

MatClient::~MatClient() {

    // Clean up sync objects
    notifyAll();
}

void MatClient::findSharedMat() {

    bool ready = false;
    while (!ready) {
        try {

            shared_memory = managed_shared_memory(open_only, shmem_name.c_str());
            shared_mat_header = shared_memory.find<shmem::SharedMatHeader>(shobj_name.c_str()).first;
            ready = shared_mat_header->ready;

        } catch (...) {
            std::cout << "Waiting for source \"" + name + "\" to start..." << std::endl;
            usleep(100000);
        } 
    }
    
    // Pass mutex to the scoped sharable_lock. This will lock the shared_mat_header->mutex
    // until wait(lock) is called.
    lock = makeLock();
    
    //std::cout << "Server found, starting." << std::endl;
    shared_mat_created = true;

    mat.create(shared_mat_header->size,
            shared_mat_header->type);

    mat.data = static_cast<uchar*>(shared_memory.get_address_from_handle(shared_mat_header->handle));
}

cv::Mat MatClient::get_shared_mat() {
    
    return mat;
}

void MatClient::notifyAll() {
    
    shared_mat_header->cond_var.notify_all();
}

void MatClient::wait() {
    
    shared_mat_header->cond_var.wait(lock);
}

void MatClient::notifyAllAndWait() {
    
    notifyAll();
    wait();
}

sharable_lock<interprocess_sharable_mutex> MatClient::makeLock(void) {
    sharable_lock<interprocess_sharable_mutex> sl(shared_mat_header->mutex);
    return sl;
}

void MatClient::set_source(const std::string source_name) {
    
    // Make sure we are not already attached to some source
    if (!shared_mat_created) {
        name = source_name;
        shmem_name = source_name + "_sh_mem";
        shobj_name = source_name + "_sh_obj";
    } else {
        std::cerr << "Cannot edit the source name because we are already reading from \"" + name + "\".";
    }
    
}

