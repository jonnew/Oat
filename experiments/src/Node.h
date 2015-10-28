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
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/sync/interprocess_semaphore.hpp>

namespace oat {
    
    namespace bip = boost::interprocess;
    using semaphore = bip::interprocess_semaphore;
    using ipvector = boost::interprocess::vector<std::shared_ptr<semaphore>>;

    enum class SinkState {
        END = -1,
        UNDEFINED = 0,
        BOUND = 1,
        ERROR = 2
    };

    class Node {
    public:

        Node() :
          sink_state_(SinkState::UNDEFINED)
        , source_ref_count_(0)           
        , mutex(1)
        , write_barrier(0)
        //, read_barrier(0)
        //, new_data_barrier(0)
        , source_read_count_(0)
        , write_number_(0) 
        { 
          // Nothing
        }

        // These operations are atomic
          
        // SINK state
        inline void set_sink_state(SinkState value) { sink_state_ = value; }
        inline SinkState sink_state(void) const { return sink_state_; }
        
        // SINK writes (~ sample number)
        inline uint64_t write_number() const { return write_number_; }
        inline uint64_t incrementWriteNumber() { return ++write_number_; }
        
        // SOURCE reference counting
        inline size_t decrementSourceRefCount() { return ++source_ref_count_; }
        ipvector::size_type decrementSourceRefCount(ipvector::size_type index) { 

            read_barrier.erase(read_barrier.begin() + index); 
            return read_barrier.size(); 
        }
        
        //inline size_t incrementSourceRefCount() { return ++source_ref_count_; }
        ipvector::size_type incrementSourceRefCount() { 
            
            std::shared_ptr<semaphore> sem (new semaphore(0));
            read_barrier.push_back(sem); 
            return read_barrier.size(); 
        }
        
        inline size_t source_ref_count(void) const { return read_barrier.size(); }
        
        // SOURCE read counting
        inline void resetSourceReadCount() { source_read_count_ = 0; }
        inline size_t incrementSourceReadCount() { return ++source_read_count_;}
        
        // Synchronization constructs
        semaphore mutex;
        semaphore write_barrier;
        ipvector read_barrier;
        //semaphore new_data_barrier;

    private:

        std::atomic<SinkState> sink_state_;      //!< SINK state 
        std::atomic<size_t> source_read_count_;  //!< Number SOURCE reads that have occured since last sink reset 
        std::atomic<size_t> source_ref_count_;   //!< Number of SOURCES sharing this
        std::atomic<uint64_t> write_number_;     //!< Number of writes to shmem that have been facilited by this node

    };

} // namespace oat


#endif	/* NODE_H */

