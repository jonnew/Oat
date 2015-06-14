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

#include <iostream>
#include <chrono>
#include <boost/interprocess/managed_shared_memory.hpp>

#include "../../lib/utility/IOFormat.h"
#include "Signals.h"
#include "SharedCVMatHeader.h"

namespace oat {

    namespace bip = boost::interprocess;

    MatServer::MatServer(const std::string& sink_name) :
      name(sink_name)
    , shmem_name(sink_name + "_sh_mem")
    , shobj_name(sink_name + "_sh_obj")
    , shsig_name(sink_name + "_sh_sig")
    , shared_object_created(false)
    , mat_header_constructed(false) {

        createSharedMat();
    }

    MatServer::MatServer(const MatServer& orig) { }

    MatServer::~MatServer() {

        notifySelf();

        // Remove_shared_memory on object destruction
        bip::shared_memory_object::remove(shmem_name.c_str());
#ifndef NDEBUG
        std::cout << oat::dbgMessage("Shared memory \'" + shmem_name + "\' was deallocated.\n");
#endif

    }

    void MatServer::createSharedMat(void) {
        
        // TODO: I am currently using a static 10 MB block to store shared
        // cv::Mat headers and data. This is a bit of a hack  until 
        // I can figure out how to resize the managed shared memory segment 
        // on the server side without causing seg faults due to bad pointers on the client side.

        try {

            // Total amount of shared memory to allocated
            size_t total_bytes = 1024e4;

            // Define shared memory
            shared_memory = bip::managed_shared_memory(bip::open_or_create,
                    shmem_name.c_str(),
                    total_bytes);

            shared_mat_header = shared_memory.find_or_construct<oat::SharedCVMatHeader>(shobj_name.c_str())();
            shared_server_state = shared_memory.find_or_construct<oat::ServerState>(shsig_name.c_str())();

        } catch (bip::interprocess_exception &ex) {
            std::cerr << ex.what() << '\n';
            exit(EXIT_FAILURE); // TODO: exit does not unwind the stack to take care of destructing shared memory objects
        }

        shared_object_created = true;
        setSharedServerState(oat::ServerRunState::RUNNING);
    }

    /**
     * Push a deep copy of cv::Mat object to shared memory along with sample number.
     * @param mat cv::Mat to push to shared memory
     * @param sample_number sample number of cv::Mat
     */
    void MatServer::pushMat(const cv::Mat& mat, const uint32_t& sample_number) {

#ifndef NDEBUG

        std::cout << oat::dbgMessage("sample: " + std::to_string(sample_number)) << "\r";
        std::cout.flush();

#endif

        // Create shared mat object if not done already
        if (!mat_header_constructed) {
            shared_mat_header->buildHeader(shared_memory, mat);
            mat_header_constructed = true;
        }

        /* START CRITICAL SECTION */
        shared_mat_header->mutex.wait();

        // Perform writes in shared memory 
        shared_mat_header->writeSample(sample_number, mat);

        // Tell each client they can proceed
        for (int i = 0; i < shared_mat_header->get_number_of_clients(); ++i) {
            shared_mat_header->read_barrier.post();
        }

        shared_mat_header->mutex.post();
        /* END CRITICAL SECTION */

        // Only wait if there is a client
        if (shared_mat_header->get_number_of_clients()) {
            shared_mat_header->write_barrier.wait();
        }

        // Tell each client they can proceed now that the write_barrier
        // has been passed
        for (int i = 0; i < shared_mat_header->get_number_of_clients(); ++i) {
            shared_mat_header->new_data_barrier.post();
        }

    }

    void MatServer::notifySelf() {

        if (shared_object_created) {
            shared_mat_header->write_barrier.post();
        }
    }
    
    void MatServer::setSharedServerState(oat::ServerRunState state) {

        if (shared_object_created) {
            shared_server_state->set_state(state);
        }
    }   

}
