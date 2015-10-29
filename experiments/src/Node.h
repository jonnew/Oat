//******************************************************************************
//* File:   SyncSharedMemoryManager.h
//* Author: Jon Newman <jpnewman snail mit dot edu>
//
//* Copyright (c) Jon Newman (jpnewman snail mit dot edu) 
//* All right reserved.
//* This file is part of the Oat project.
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

#ifndef NODE_H
#define	NODE_H

#include <atomic>
#include <boost/shared_ptr.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/smart_ptr/unique_ptr.hpp>
#include <boost/interprocess/smart_ptr/shared_ptr.hpp>
#include <boost/interprocess/smart_ptr/deleter.hpp>
#include <boost/interprocess/sync/interprocess_semaphore.hpp>

namespace oat {

    namespace bip = boost::interprocess;
    using semaphore = bip::interprocess_semaphore;
    
//    using unique_ptr_type = 
//            bip::managed_unique_ptr<
//              semaphore
//            , bip::managed_shared_memory
//            >::type;
//    
//    using unique_ptr_vector_t = 
//            bip::vector< 
//              unique_ptr_type
//            , bip::allocator<unique_ptr_type, bip::managed_shared_memory::segment_manager> 
//            >;
    
    using void_allocator_t = bip::allocator<void, bip::managed_shared_memory::segment_manager> ;
    using segment_manager_t = bip::managed_shared_memory::segment_manager;
    using deleter_t = bip::deleter<semaphore, segment_manager_t>;
    using shared_ptr_t = bip::shared_ptr < semaphore
                                         , void_allocator_t 
                                         , deleter_t
                                         >;
    using shared_ptr_allocator_t = bip::allocator<shared_ptr_t, bip::managed_shared_memory::segment_manager> ;
    using shared_ptr_vector_t = bip::vector < shared_ptr_t
                                            , shared_ptr_allocator_t 
                                            >;
    
    enum class SinkState {
        END = -1,
        UNDEFINED = 0,
        BOUND = 1,
        ERROR = 2
    };

    // TODO : or, make Node a class template with template parameter determining
    // the data type.
//    enum class NodeType {
//        UNDEFINED = 0,
//        FRAME = 1,
//        POSITION = 2,
//    };

    class Node {
    public:

        Node()    
        {
//            // Populate semaphore array
//            for (size_t i = 0; i < read_barrier.size(); i ++) {
//                ip_shared_ptr<semaphore> sem(new semaphore(0));
//                read_barrier[i] = sem;
//            }
        }

        // These operations are atomic

        // SINK state
        void set_sink_state(SinkState value) { sink_state_ = value; }
        SinkState sink_state(void) const { return sink_state_; }

        // SINK writes (~ sample number)
        uint64_t write_number() const { return write_number_; }
        uint64_t incrementWriteNumber() { return ++write_number_; }

        // SOURCE reference counting
        //size_t decrementSourceRefCount() { return --source_ref_count_; }

//        ipvector::size_type decrementSourceRefCount(ipvector::size_type index) {
//
//            read_barrier.erase(read_barrier.begin() + index);
//            return read_barrier.size();
//        }

        //size_t incrementSourceRefCount() { return source_ref_count_++; }

        void getReadBarrier(const char* name, bip::managed_shared_memory& shmem) {
            read_barrier = shmem.find_or_construct<shared_ptr_vector_t>(name)(shmem.get_segment_manager());
        }
        
        shared_ptr_vector_t::size_type incrementSourceRefCount(bip::managed_shared_memory& shmem) {

            shared_ptr_t p(bip::make_managed_shared_ptr(shmem.construct<semaphore>(bip::anonymous_instance)(0), shmem));


            read_barrier->push_back(boost::move(p));
            source_ref_count_ ++;
            return (read_barrier->size() - 1); // Return the index of the semaphore
        }

        size_t source_ref_count(void) const { return source_ref_count_; }

        // SOURCE read counting

        void resetSourceReadCount() { source_read_count_ = 0; }

        size_t incrementSourceReadCount() { return ++source_read_count_; }

        // Synchronization constructs
        semaphore mutex {1};
        semaphore write_barrier {0};
        shared_ptr_vector_t * read_barrier;
        //semaphore new_data_barrier;

    private:

        std::atomic<SinkState> sink_state_ {oat::SinkState::UNDEFINED}; //!< SINK state 
        std::atomic<size_t> source_read_count_ {0}; //!< Number SOURCE reads that have occured since last sink reset 
        std::atomic<size_t> source_ref_count_ {0}; //!< Number of SOURCES sharing this
        std::atomic<uint64_t> write_number_ {0}; //!< Number of writes to shmem that have been facilited by this node

    };

} // namespace oat

#endif	/* NODE_H */

