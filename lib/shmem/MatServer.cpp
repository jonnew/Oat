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

#include <deque>
#include <thread>
#include <boost/interprocess/managed_shared_memory.hpp>
#include "SharedCVMatHeader.h"
#include "SharedCVMatHeader.cpp" // TODO: Why???

using namespace boost::interprocess;

MatServer::MatServer(const std::string server_name) :
  name(server_name)
, shmem_name(name + "_sh_mem")
, shobj_name(name + "_sh_obj")
, shared_object_created(false)
{ 
	// Start the server thread
	server_thread = std::thread(&MatServer::syncServeFromBuffer, this, this);	
}

MatServer::MatServer(const MatServer& orig) { }

MatServer::~MatServer() {

	// Jon the server thread back with the main one
	server_thread.join();
		
    // Remove_shared_memory on object destruction
    shared_mat_header->new_data_condition.notify_all();
    shared_memory_object::remove(shmem_name.c_str());
}

void MatServer::createSharedMat(cv::Mat model) {
    
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
}

void MatServer::set_shared_mat(cv::Mat mat) {

    if (!shared_object_created) {
        createSharedMat(mat); 
    }
    
    // Exclusive scoped_lock on the shared_mat_header->mutex
    scoped_lock<interprocess_sharable_mutex> lock(shared_mat_header->mutex);
    
    // Perform write in shared memory
    //memcpy(shared_mat_data_ptr, mat.data, data_size);
    shared_mat_header->set_value(mat);
    
    // Notify all client processes they can now access the data
    shared_mat_header->new_data_condition.notify_all();
    
} // Lock is released on scope exit

void MatServer::set_name(const std::string sink_name) {
    
    // Make sure we are not already attached to some source
    if (!shared_object_created) {
        name = sink_name;
        shmem_name = sink_name + "_sh_mem";
        shobj_name = sink_name + "_sh_obj";
    } else {
        std::cerr << "Cannot edit the sink name because we are already reading from \"" + name + "\".";
    }
}
