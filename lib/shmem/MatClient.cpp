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
#include <boost/thread/thread_time.hpp>
#include <boost/thread.hpp>

namespace oat {

     namespace bip = boost::interprocess;

    MatClient::MatClient(const std::string source_name) :
      name(source_name)
    , shmem_name(source_name + "_sh_mem")
    , shobj_name(source_name + "_sh_obj")
    , shsig_name(source_name + "_sh_sig")
    , shared_object_found(false)
    , mat_attached_to_header(false)
    , read_barrier_passed(false) {

        findSharedMat();
    }

    MatClient::~MatClient() {

        detachFromShmem();
    }

    /**
     * Use shmem_name and shobj_name to find (or create) the requested shared cvmat
     */
    void MatClient::findSharedMat() {


        // TODO: I am currently using a static 10 MB block to store shared
        // cv::Mat headers and data. This is a bit of a hack  until 
        // I can figure out how to resize the managed shared memory segment 
        // on the server side without causing seg faults due to bad pointers on the client side.
        // If the client creates the shared memory, it does not allocate room for the cv::Mat data
        // The server will need to resize the shared memory to make room.
        //size_t total_bytes = sizeof (oat::SharedCVMatHeader) + 1024; 
        
        try {

            size_t total_bytes = 1024e4;

            shared_memory = bip::managed_shared_memory(bip::open_or_create, shmem_name.c_str(), total_bytes);
            shared_mat_header = shared_memory.find_or_construct<oat::SharedCVMatHeader>(shobj_name.c_str())();
            shared_server_state = shared_memory.find_or_construct<oat::ServerState>(shsig_name.c_str())();
            

        } catch (bip::interprocess_exception& ex) {
            std::cerr << ex.what() << '\n';
            exit(EXIT_FAILURE); // TODO: exit does not unwind the stack to take care of destructing shared memory objects
        }

        shared_object_found = true;
        
        // Make sure everyone using this shared memory knows that another client
        // has joined
        number_of_clients = shared_mat_header->incrementClientCount();
    }

    /**
     * Get the cv::Mat object from shared memory
     * @param value The cv::Mat object to be copied from shared memory
     * @return True if the result is (1) valid and (2) successfully obeyed all 
     * interprocess synchronization mechanisms. False if there were timeouts during
     * wait() calls, meaning that the cv::Mat objects has possibly not been assigned
     * or that proceeding to process in a loop that will recall this function may result
     * in server/client desynchronization.
     */
    bool MatClient::getSharedMat(cv::Mat& value) {

        boost::system_time timeout =
                boost::get_system_time() + boost::posix_time::milliseconds(10);

        try {
            if (!read_barrier_passed) {

                if (!shared_mat_header->read_barrier.timed_wait(timeout)) {
                    return false;
                }

                // Write down that we made it past the read_barrier in case the
                // new_data_barrier times out.
                read_barrier_passed = true;

                /* START CRITICAL SECTION */
                shared_mat_header->mutex.wait();

                if (!mat_attached_to_header) {
                    // Cannot do this until the server has called build header, which is 
                    // why it is here, instead of in constructor
                    shared_mat_header->attachMatToHeader(shared_memory, shared_cvmat);
                    mat_attached_to_header = true;
                }

                // Assign the latest cv::Mat and get its timestamp and write index
                value = shared_cvmat.clone(); // TODO: this clone might not be nessesary (http://docs.opencv.org/modules/core/doc/intro.html#multi-threading-and-re-enterability)
                current_sample_number = shared_mat_header->get_sample_number();

                // Now that this client has finished its read, update the count
                shared_mat_header->client_read_count++;

                // If all clients have read, signal the barrier
                if (shared_mat_header->client_read_count == shared_mat_header->get_number_of_clients()) {
                    shared_mat_header->write_barrier.post();
                    shared_mat_header->client_read_count = 0;
                }

                shared_mat_header->mutex.post();
                /* END CRITICAL SECTION */
            }

            if (!shared_mat_header->new_data_barrier.timed_wait(timeout)) {
                return false;
            }

            read_barrier_passed = false;
            return true; // Result is valid and all waits have operated without timeout

        } catch (bip::interprocess_exception ex) {
            
            // Something went wrong during shmem access so result is invalid
            // Usually due to SIGINT being called during read_barrier timed
            // wait
            return false;
        }
    }
    
    oat::ServerRunState MatClient::getServerRunState() {
        
        return shared_server_state->get_state();
    }

    void MatClient::detachFromShmem() {

        if (shared_object_found) {

            // Make sure nobody is going to wait on a disposed object
            number_of_clients = shared_mat_header->decrementClientCount();

#ifndef NDEBUG
            std::cout << "Number of clients in \'" + shmem_name + "\' was decremented.\n";
#endif

        }
    }

} // namespace shmem
