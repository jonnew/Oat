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
cli_name(source_name)
, shmem_name(source_name + "_sh_mem")
, shobj_name(source_name + "_sh_obj") {
}

MatClient::MatClient(const MatClient& orig) {
}

MatClient::~MatClient() {

    // Clean up sync objects
    cli_shared_mat_header->cond_var.notify_all();
}

void MatClient::findSharedMat() {

    while (!cli_shared_mat_header->ready) {
        try {

            shared_memory = managed_shared_memory(open_only, shmem_name.c_str());
            cli_shared_mat_header = shared_memory.find<shmem::SharedMatHeader>(shobj_name.c_str()).first;

        } catch (...) {
            std::cout << "Waiting for source \"" + cli_name + "\" to start..." << std::endl;
            usleep(100000);
        } 
    }
    
    std::cout << "Server found, starting." << std::endl;
    cli_shared_mat_created = true;

    mat.create(cli_shared_mat_header->size,
            cli_shared_mat_header->type);

    mat.data = static_cast<uchar*>(shared_memory.get_address_from_handle(cli_shared_mat_header->handle));

}

cv::Mat MatClient::get_shared_mat() {

	// TODO: This should be wrapped in the lock mechanism, right? 
	// This can be separate from wait(), which can be up to the user
	// to call. Same for the write side, probably.
    return mat;
}
