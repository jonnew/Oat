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

#ifndef SMSERVER_H
#define	SMSERVER_H

#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/lockfree/spsc_queue.hpp>

#include "SyncSharedMemoryObject.h"

namespace bip = boost::interprocess;

namespace shmem {

    template<class T, template <typename> class SharedMemType = shmem::SyncSharedMemoryObject>
    class SMServer {
    public:
        SMServer(std::string sink_name);
        SMServer(const SMServer& orig);
        virtual ~SMServer();

        void createSharedObject(void);
        void pushObject(T value);

    private:

        // Name of this server
        std::string name;

        // Buffer
        boost::lockfree::spsc_queue<T, boost::lockfree::capacity<1024> > buffer;
        
        // Server threading
        std::thread server_thread;
        std::mutex server_mutex;
        std::condition_variable serve_condition;
        bool running; // Server running

        // Shared memory and managed object names
        SharedMemType<T>* shared_object; // Defaults to shmem::SyncSharedMemoryObject<T>
        std::string shmem_name, shobj_name;
        bip::managed_shared_memory shared_memory;
        bool shared_object_created;

        void createSharedObject(size_t bytes);
        void serveFromBuffer(void);

    };

    template<class T, template <typename> class SharedMemType>
    SMServer<T, SharedMemType>::SMServer(std::string sink_name) :
      name(sink_name)
    , shmem_name(sink_name + "_sh_mem")
    , shobj_name(sink_name + "_sh_obj")
    , shared_object_created(false)
    , running(true) {

        // Start the server thread
        server_thread = std::thread(&SMServer<T, SharedMemType>::serveFromBuffer, this);
    }

    template<class T, template <typename> class SharedMemType>
    SMServer<T, SharedMemType>::SMServer(const SMServer<T, SharedMemType>& orig) {
    }

    template<class T, template <typename> class SharedMemType>
    SMServer<T, SharedMemType>::~SMServer() {

        // Remove_shared_memory on object destruction
        shared_object->new_data_condition.notify_all();
        bip::shared_memory_object::remove(shmem_name.c_str());
    }

    template<class T, template <typename> class SharedMemType>
    void SMServer<T, SharedMemType>::createSharedObject() {

        try {

            // Clean up any potential leftovers
            bip::shared_memory_object::remove(shmem_name.c_str());

            // Allocate shared memory
            shared_memory = bip::managed_shared_memory(
                    bip::open_or_create, 
                    shmem_name.c_str(), 
                    sizeof (SharedMemType<T>) + 1024);

            // Make the shared object
            shared_object = shared_memory.find_or_construct<SharedMemType < T>> (shobj_name.c_str())();

            // Set the ready flag
            shared_object_created = true;

        } catch (bip::bad_alloc &ex) {
            std::cerr << ex.what() << '\n';
        }
    }

    template<class T, template <typename> class SharedMemType>
    void SMServer<T, SharedMemType>::pushObject(T value) {

        // Push data onto ring buffer
        buffer.push(value);

        // notify server thread that data is available
        serve_condition.notify_one();

    }

    template<class T, template <typename> class SharedMemType>
    void SMServer<T, SharedMemType>::serveFromBuffer() {

        while (running) {

            // Proceed only if mat_buffer has data
            std::unique_lock<std::mutex> lk(server_mutex);
            serve_condition.wait(lk); 

            T value;
            while (buffer.pop(value)) {

                if (!shared_object_created) {
                    createSharedObject();
                }
                // Exclusive scoped_lock on the shared_mat_header->mutex
                bip::scoped_lock<bip::interprocess_sharable_mutex>
                        lock(shared_object->mutex);

                // Perform write in shared memory
                shared_object->set_value(value);

                // Notify all client processes they can now access the data
                shared_object->new_data_condition.notify_all();
                shared_object->new_data_condition.wait(lock);
            }
        } // Scoped lock is released
    }

} // namespace shmem 

#endif	/* SMSERVER_H */