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

#ifndef NODEMANAGER_H
#define	NODEMANAGER_H

#include <atomic>

namespace oat {
    
    namespace bip = boost::interprocess;

    enum class SinkState {
        END = -1,
        UNDEFINED = 0,
        BOUND = 1,
        ERROR = 2
    };

    class NodeManger {
    public:

        NodeManger() :
          sink_state_(SinkState::UNDEFINED)
        , source_ref_count_(0)           
        , mutex(1)
        , write_barrier(0)
        , read_barrier(0)
        //, new_data_barrier(0)
        , source_read_count(0)
        , write_number_(0) 
        { 
          // Nothing
        }

        // These operations are atomic
          
        // SINK state
        void set_server_state(SinkState value) { sink_state_ = value; }
        SinkState get_server_state(void) const { return sink_state_; }
        
        // SINK writes (~ sample number)
        uint64_t write_number() const { return write_number; }
        uint64_t incrementWriteNumber() { return ++write_number_; }
        
        // SOURCE reference counting
        size_t decrementSourceRefCount() { return --source_ref_count_; }
        size_t incrementSourceRefCount() { return ++source_ref_count_; }
        size_t source_ref_count(void) const { return source_ref_count_; }
        void resetSourceReadCount() { source_ref_count_ = 0; }
        size_t incrementSourceReadCount() { return ++source_read_count;}
        
        // Synchronization constructs
        bip::interprocess_semaphore mutex;
        bip::interprocess_semaphore write_barrier;
        bip::interprocess_semaphore read_barrier;
        //bip::interprocess_semaphore new_data_barrier;

        
    private:

        std::atomic<SinkState> sink_state_;     //!< SINK state 
        std::atomic<size_t> source_read_count;  //!< Number SOURCE reads that have occured since last sink reset 
        std::atomic<size_t> source_ref_count_;  //!< Number of SOURCES sharing this
        std::atomic<uint64_t> write_number_;    //!< Number of writes to shmem that have been facilited by this node

    };

} // namespace oat


#endif	/* NODEMANAGER_H */

