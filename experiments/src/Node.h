//******************************************************************************
//* File:   Node.h
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
#include <boost/interprocess/sync/interprocess_semaphore.hpp>

namespace oat {

    using semaphore = boost::interprocess::interprocess_semaphore;
    
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
            // Nothing
        }

        // These operations are atomic

        // SINK state
        void set_sink_state(SinkState value) { sink_state_ = value; }
        SinkState sink_state(void) const { return sink_state_; }

        // SINK writes (~ sample number)
        uint64_t write_number() const { return write_number_; }
        uint64_t incrementWriteNumber() { return ++write_number_; }

        // SOURCE read counting
        size_t incrementSourceReadCount() { ++source_read_count_; }
        void resetSourceReadCount() { source_read_count_ = 0; }

        // SOURCE reference counting
        size_t decrementSourceRefCount() { return --source_ref_count_; }
        size_t incrementSourceRefCount() { 
            if (source_ref_count_ >= 10)
                throw std::runtime_error("Maximum of 10 SOURCEs can be bound to a node.");
            return ++source_ref_count_; 
        }
        size_t source_ref_count(void) const { return source_ref_count_; }

        // Synchronization constructs
        semaphore write_barrier {0};
        
        // Dead simple and static, the suckless way
        semaphore& readBarrier(size_t index) {
            switch (index) {
                case 0:
                    return rb0_;
                    break;
                case 1:
                    return rb1_;
                    break;
                case 2:
                    return rb2_;
                    break;    
                case 3:
                    return rb3_;
                    break;    
                case 4:
                    return rb4_;
                    break;
                case 5:
                    return rb5_;
                    break;
                case 6:
                    return rb6_;
                    break;
                case 7:
                    return rb7_;
                    break;    
                case 8:
                    return rb8_;
                    break;    
                case 9:
                    return rb9_;
                    break;
                default:
                    throw std::runtime_error("Source index out of range.");
                    break;
            } 
        }
        
    private:

        std::atomic<SinkState> sink_state_ {oat::SinkState::UNDEFINED}; //!< SINK state 
        std::atomic<size_t> source_read_count_ {0}; //!< Number SOURCE reads that have occured since last sink reset 
        std::atomic<size_t> source_ref_count_ {0}; //!< Number of SOURCES sharing this
        std::atomic<uint64_t> write_number_ {0}; //!< Number of writes to shmem that have been facilited by this node
        
        // 10 sources max
        semaphore rb0_ {0}, rb1_ {0}, rb2_ {0}, rb3_ {0}, rb4_ {0}, 
                  rb5_ {0}, rb6_ {0}, rb7_ {0}, rb8_ {0}, rb9_ {0};
    };

} // namespace oat

#endif	/* NODE_H */

