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

#ifndef SMCLIENT_H
#define	SMCLIENT_H

#include <string>
#include <boost/thread/thread_time.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>

#include "../../lib/utility/IOFormat.h"
#include "SyncSharedMemoryObject.h"
#include "SharedMemoryManager.h"

namespace oat {

    namespace bip = boost::interprocess;

    template<class T, template <typename IOType> class SharedMemType = oat::SyncSharedMemoryObject>
    class SMClient {
    public:
        SMClient(std::string source_name);
        SMClient(const SMClient& orig);
        virtual ~SMClient();

        /**
         * Get the object from shared memory
         * @param value Reference to which assign the shared object
         * @return true if assignment was successful, false if assignment timed out.
         */
        bool getSharedObject(T& value);
        
        oat::ServerRunState getSourceRunState(void);

        /**
         * Get the current sample number
         * @return current sample
         */
        uint32_t get_current_time_stamp(void) { return current_time_stamp; }

    private:

        SharedMemType<T>* shared_object; // Defaults to oat::SyncSharedMemoryObject<T>
        oat::SharedMemoryManager* shared_mem_manager;
        std::string name;
        std::string shmem_name, shobj_name, shmgr_name;
        bool shared_object_found;
        bool read_barrier_passed;
        bip::managed_shared_memory shared_memory;

        // Number of clients, including *this, attached to the shared memory indicated
        // by shmem_name
        size_t number_of_clients;
        
        // Time keeping
        uint32_t current_time_stamp;

        // Find shared object in shmem
        int findSharedObject(void);

        // Decrement the number of clients in shmem
        void detachFromShmem(void);
    };

    template<class T, template <typename> class SharedMemType>
    SMClient<T, SharedMemType>::SMClient(std::string source_name) :
      name(source_name)
    , shmem_name(source_name + "_sh_mem")
    , shobj_name(source_name + "_sh_obj")
    , shmgr_name(source_name + "_sh_mgr")
    , shared_object_found(false)
    , read_barrier_passed(false)
    , current_time_stamp(0) {

        findSharedObject();
    }

    template<class T, template <typename> class SharedMemType>
    SMClient<T, SharedMemType>::~SMClient() {

        detachFromShmem();
    }

    template<class T, template <typename> class SharedMemType>
    int SMClient<T, SharedMemType>::findSharedObject() {

        int client_num;

        try {

            // Allocate shared memory
            shared_memory = bip::managed_shared_memory(
                    bip::open_or_create,
                    shmem_name.c_str(),
                    sizeof(SharedMemType<T>) + sizeof(oat::SharedMemoryManager) + 1024);

            // Find the object in shared memory
            shared_object = shared_memory.find_or_construct<SharedMemType < T >> (shobj_name.c_str())();
            shared_mem_manager = shared_memory.find_or_construct<oat::SharedMemoryManager>(shmgr_name.c_str())();
            shared_object_found = true;

        } catch (bip::interprocess_exception& ex) {
            std::cerr << ex.what() << '\n';
            exit(EXIT_FAILURE); // TODO: exit does not unwind the stack to take care of destructing shared memory objects
        }

        shared_object_found = true;
        
        // Make sure everyone using this shared memory knows that another client
        // has joined
        number_of_clients = shared_mem_manager->incrementClientRefCount();

        return number_of_clients;
    }

    /**
     * Get the object from shared memory.
     * @param value The object to be copied from shared memory.
     * @return True if the result is (1) valid and (2) successfully obeyed all 
     * interprocess synchronization mechanisms. False if there were timeouts during
     * wait() calls, meaning that the value has possibly not been assigned
     * or that proceeding to process in a loop that will recall this function may result
     * in server/client desynchronization.
     */
    template<class T, template <typename> class SharedMemType>
    bool SMClient<T, SharedMemType>::getSharedObject(T& value) {

        boost::system_time timeout =
                boost::get_system_time() + boost::posix_time::milliseconds(10);
        
        try {
            
            if (!read_barrier_passed) {

                if (!shared_object->read_barrier.timed_wait(timeout)) {
                    return false;
                }

                // Write down that we made it past the read_barrier in case the
                // new_data_barrier times out.
                read_barrier_passed = true;

                /* START CRITICAL SECTION */
                shared_object->mutex.wait();

                value = shared_object->get_value();
                current_time_stamp = shared_object->get_sample_number();

                // Now that this client has finished its read, update the count
                shared_object->client_read_count++;

                // If all clients have read, signal the write barrier
                if (shared_object->client_read_count >= shared_mem_manager->get_client_ref_count()) {
                    shared_object->write_barrier.post();
                    shared_object->client_read_count = 0;
                }

                shared_object->mutex.post();
                /* END CRITICAL SECTION */
            }

            if (!shared_object->new_data_barrier.timed_wait(timeout)) {
                return false;
            }

            // Reset the read_barrier_passed switch
            read_barrier_passed = false;
            return true;
            
        } catch (bip::interprocess_exception ex) {

            // Something went wrong during shmem access so result is invalid
            // Usually due to SIGINT being called during read_barrier timed
            // wait
            return false;
        }
    }
    
    template<class T, template <typename> class SharedMemType>
    oat::ServerRunState SMClient<T, SharedMemType>::getSourceRunState() {
        
        if (shared_object_found) 
            return shared_mem_manager->get_server_state();
        else
            return oat::ServerRunState::UNDEFINED;
    }

    template<class T, template <typename> class SharedMemType>
    void SMClient<T, SharedMemType>::detachFromShmem() {

        if (shared_object_found) {

            // Make sure nobody is going to wait on a disposed object
            number_of_clients = shared_mem_manager->decrementClientRefCount();

            // If the client reference count is 0 and there is no server 
            // attached to the shared mat, deallocate the shmem
            if (number_of_clients == 0 && shared_mem_manager->get_server_state() == oat::ServerRunState::END) {

                // Ensure that no server is deadlocked
                shared_object->write_barrier.post();
                
                bip::shared_memory_object::remove(shmem_name.c_str());
#ifndef NDEBUG
                std::cout << oat::dbgMessage("Shared memory \'" + shmem_name + "\' was deallocated.\n");
#endif
            } else {
#ifndef NDEBUG
                std::cout << oat::dbgMessage("Number of clients in \'" + shmem_name + "\' was decremented.\n");
#endif
            }
        }
    }
} // namespace shmem 

#endif	/* SMCLIENT_H */

