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
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/sharable_lock.hpp>
#include <boost/interprocess/sync/interprocess_sharable_mutex.hpp>

#include "SyncSharedMemoryObject.h"

namespace bip = boost::interprocess;

namespace shmem {

    template<class T, template <typename IOType> class SharedMemType = shmem::SyncSharedMemoryObject>
    class SMClient {
    public:
        SMClient(std::string source_name);
        SMClient(const SMClient& orig);
        virtual ~SMClient();

        T get_value(void);
        void notifyAndWait(void);
        void notifySelf(void);

    private:

        SharedMemType<T>* shared_object; // Defaults to shmem::SyncSharedMemoryObject<T>

        std::string name, shmem_name, shobj_name;
        bool shared_object_found,  terminated;
        bip::managed_shared_memory cli_shared_memory;
        bip::sharable_lock<bip::interprocess_sharable_mutex> lock;
        bip::sharable_lock<bip::interprocess_sharable_mutex> makeLock();

        void findSharedObject(void);

    };

    template<class T, template <typename> class SharedMemType>
    SMClient<T, SharedMemType>::SMClient(std::string source_name) :
      name(source_name)
    , shmem_name(source_name + "_sh_mem")
    , shobj_name(source_name + "_sh_obj") 
    , shared_object_found(false)
    , terminated(false) { }

    template<class T, template <typename> class SharedMemType>
    SMClient<T, SharedMemType>::SMClient(const SMClient<T, SharedMemType>& orig) {
    }

    template<class T, template <typename> class SharedMemType>
    SMClient<T, SharedMemType>::~SMClient() {

        // Clean up sync objects
        shared_object->new_data_condition.notify_all();
    }

    template<class T, template <typename> class SharedMemType>
    void SMClient<T, SharedMemType>::findSharedObject() {

        // TODO: This should be replaced some type of formal handshake between
        // server and client.
        bip::shared_memory_object::remove(shmem_name.c_str());
        
        while (!shared_object_found) {
            try {

                // Allocate shared memory
                cli_shared_memory = bip::managed_shared_memory(bip::open_only, shmem_name.c_str());

                // Find the object in shared memory
                shared_object = cli_shared_memory.find<SharedMemType < T >> (shobj_name.c_str()).first;
                shared_object_found = true;

            } catch (bip::interprocess_exception& e) {
                usleep(100000); // Wait for shared memory to be created by SOURCE
            }

            if (terminated)
                exit(EXIT_FAILURE); // Nothing to clean, so we are OK to exit.
        }
        
        lock = makeLock();
    }

    template<class T, template <typename> class SharedMemType>
    T SMClient<T, SharedMemType>::get_value() {

        if (!shared_object_found) {
            findSharedObject(); // Creates lock targeting shared_object->mutex, and engages
        }

        // Before reading data, we must be notified by the server that the 
        // data is valid.
        //shared_object->new_data_condition.wait(lock);
        return shared_object->get_value();
    }

    template<class T, template <typename> class SharedMemType>
    bip::sharable_lock<bip::interprocess_sharable_mutex> SMClient<T, SharedMemType>::makeLock(void) {
        bip::sharable_lock<bip::interprocess_sharable_mutex> sl(shared_object->mutex); // defer_lock
        return sl;
    }

    template<class T, template <typename> class SharedMemType>
    void SMClient<T, SharedMemType>::notifyAndWait() {

        // Wait for notification from SOURCE to grant access to shared memory
        shared_object->new_data_condition.notify_all();
        shared_object->new_data_condition.wait(lock);
    }

    template<class T, template <typename> class SharedMemType>
    void SMClient<T, SharedMemType>::notifySelf() {

        if (shared_object_found) {
            shared_object->new_data_condition.notify_one();
        }
        terminated = true;
    }
    
} // namespace shmem 

#endif	/* SMCLIENT_H */

