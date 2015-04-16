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

    template<class T, template <typename IOType> class SharedMemType> // = shmem::SyncSharedMemoryObject
    class SMClient {
    public:
        SMClient(std::string source_name);
        SMClient(const SMClient& orig);
        virtual ~SMClient();

        T get_value(void);

    private:

        SharedMemType<T>* shared_object; // Defaults to shmem::SyncSharedMemoryObject<T>

        std::string name, shmem_name, shobj_name;
        bool shared_object_found = false;
        bip::managed_shared_memory cli_shared_memory;
        bip::sharable_lock<bip::interprocess_sharable_mutex> lock;
        bip::sharable_lock<bip::interprocess_sharable_mutex> makeLock();

        void findSharedObject(void);

    };

    template<class T, template <typename> class SharedMemType>
    SMClient<T, SharedMemType>::SMClient(std::string source_name) :
    name(source_name)
    , shmem_name(source_name + "_sh_mem")
    , shobj_name(source_name + "_sh_obj") {
    }

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

        try {

            // Allocate shared memory
            cli_shared_memory = bip::managed_shared_memory(bip::open_only, shmem_name.c_str());

            // Find the object in shared memory
            shared_object = cli_shared_memory.find<SharedMemType < T >> (shobj_name.c_str()).first;
            shared_object_found = true;

        } catch (bip::interprocess_exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
            std::cerr << "  This is likely due to the SOURCE, \"" << name << "\", not being started.\n";
            std::cerr << "  Did you start the SOURCE, \"" << name << "\", before staring this client?" << std::endl;
            exit(EXIT_FAILURE); // TODO: exit does not unwind the stack to take care of destructing shared memory objects
        }

        lock = makeLock();
    }

    template<class T, template <typename> class SharedMemType>
    T SMClient<T, SharedMemType>::get_value() {

        return shared_object->get_value();
    }

    template<class T, template <typename> class SharedMemType>
    bip::sharable_lock<bip::interprocess_sharable_mutex> SMClient<T, SharedMemType>::makeLock(void) {
        bip::sharable_lock<bip::interprocess_sharable_mutex> sl(shared_object->mutex); // defer_lock
        return sl;
    }

} // namespace shmem 

#endif	/* SMCLIENT_H */

