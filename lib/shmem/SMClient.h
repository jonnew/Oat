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
#include <boost/chrono.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>

#include "SyncSharedMemoryObject.h"

namespace shmem {
    
    namespace bip = boost::interprocess;

    template<class T, template <typename IOType> class SharedMemType = shmem::SyncSharedMemoryObject>
    class SMClient {
    public:
        SMClient(std::string source_name);
        SMClient(const SMClient& orig);
        virtual ~SMClient();

        // Find shared object
        int findSharedObject(void);
        
        // Read the object value
        void get_value(T& value);
        
        // Auto notification to exit any lingering wait() calls
        void notifySelf(void);

    private:

        SharedMemType<T>* shared_object; // Defaults to shmem::SyncSharedMemoryObject<T>

        std::string name;
        std::string shmem_name, shobj_name;
        bool shared_object_found;
        bip::managed_shared_memory shared_memory;
        
        void detachFromShmem(void);
    };

    template<class T, template <typename> class SharedMemType>
    SMClient<T, SharedMemType>::SMClient(std::string source_name) :
    name(source_name)
    , shmem_name(source_name + "_sh_mem")
    , shobj_name(source_name + "_sh_obj")
    , shared_object_found(false) {  }

    template<class T, template <typename> class SharedMemType>
    SMClient<T, SharedMemType>::SMClient(const SMClient<T, SharedMemType>& orig) {
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
                    sizeof (SharedMemType<T>) + 1024);

            // Find the object in shared memory
            shared_object = shared_memory.find_or_construct<SharedMemType < T >> (shobj_name.c_str())();
            shared_object_found = true;

        } catch (bip::interprocess_exception& ex) {
            std::cerr << ex.what() << '\n';
            exit(EXIT_FAILURE); // TODO: exit does not unwind the stack to take care of destructing shared memory objects
        }

        // Make sure everyone using this shared memory knows that another client
        // has joined
        shared_object->mutex.wait();
        shared_object->number_of_clients++;
        client_num = shared_object->number_of_clients;
        shared_object->mutex.post();
        
        return client_num;
    }

    template<class T, template <typename> class SharedMemType>
    void SMClient<T, SharedMemType>::get_value(T& value) {
        
        shared_object->read_barrier.wait();
        
        /* START CRITICAL SECTION */
        shared_object->mutex.wait();

        value = shared_object->get_value();

        // Now that this client has finished its read, update the count
        shared_object->client_read_count++;

        // If all clients have read, signal the write barrier
        if (shared_object->client_read_count == shared_object->number_of_clients) {
            shared_object->write_barrier.post();
            shared_object->client_read_count = 0;
        }

        shared_object->mutex.post();
        /* END CRITICAL SECTION */
        
        shared_object->new_data_barrier.wait();
             
    }

    template<class T, template <typename> class SharedMemType>
    void SMClient<T, SharedMemType>::detachFromShmem() {
        
        if (shared_object_found) {

            // Make sure nobody is going to wait on a disposed object
            shared_object->mutex.wait();
            shared_object->number_of_clients--;
            shared_object->mutex.post();

#ifndef NDEBUG
            std::cout << "Number of clients in \'" + shmem_name + "\' was decremented.\n";
#endif

        }
    }
    template<class T, template <typename> class SharedMemType>
    void SMClient<T, SharedMemType>::notifySelf() {

        if (shared_object_found) {
            shared_object->read_barrier.post();
            shared_object->new_data_barrier.wait();
        }
    }
} // namespace shmem 

#endif	/* SMCLIENT_H */

