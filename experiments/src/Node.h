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

#ifndef OAT_NODE_H
#define	OAT_NODE_H

#include <array>
#include <atomic>
#include <bitset>
#include <string>
#include <boost/interprocess/sync/interprocess_semaphore.hpp>

#include "ForwardsDecl.h"

namespace oat {

enum class SinkState {
    END = -1,
    UNDEFINED = 0,
    BOUND = 1,
    ERROR = 2
};

class Node {
public:

    using semaphore = bip::interprocess_semaphore;

    Node()
    {
        source_slots_.reset();
    }

    // SINK state
    inline void set_sink_state(SinkState value) { sink_state_ = value; }
    inline SinkState sink_state(void) const { return sink_state_; }

    // SINK writes (~ sample number)
    inline uint64_t write_number() const { return write_number_; }
    inline uint64_t incrementWriteNumber() { return ++write_number_; }

    // SOURCE read counting
    inline size_t incrementSourceReadCount() { return ++source_read_count_; }
    inline void resetSourceReadCount() { source_read_count_ = 0; }

    // SOURCE slots
    static constexpr size_t NUM_SLOTS {10};
    size_t acquireSlot() {

        mutex_.wait();

        if (source_slots_.all()) {
            mutex_.post();
            throw std::runtime_error("Maximum of " + std::to_string(NUM_SLOTS)
                    + " SOURCEs can be bound to a node.");
        }

        size_t index = 0;
        while (source_slots_[index])
            ++index;

        source_slots_[index] = true;
        source_ref_count_ = source_slots_.count();

        mutex_.post();

        return index;
    }

    size_t releaseSlot(size_t index) {

        if (index >= source_slots_.size())
            throw std::runtime_error("Index out of bounds.");

        mutex_.wait();
        source_slots_[index] = false;
        source_ref_count_ = source_slots_.count();
        mutex_.post();

        return source_ref_count_;
    }

    inline size_t source_ref_count(void) const { return source_ref_count_; }

    // Synchronization constructs
    // write _always_ occurs before read. By starting at 1, the writer is not
    // blocked by an initial wait. Readers to do not post to the write_barrier
    // until a write occurs.
    semaphore write_barrier {1};

    // This method is required because an std::array of semaphores requires
    // each semaphore to be copy-constructed to intialized the array.
    // Because of thier nature, semaphores are NOT copy constructable, so
    // this approach does not work.
    semaphore& read_barrier(size_t index) {

        if (!source_slots_[index])
            throw std::runtime_error("Requested index refers to a SOURCE that is not bound to this node.");

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

    // Notify write finished
    // Post to each filled source slot
    void notifySources() {

        // Naive but probably not worth optimizing
        for (size_t i = 0; i < source_slots_.size(); i++)
            if (source_slots_[i])
                read_barrier(i).post();
    }

private:

    std::atomic<SinkState> sink_state_ {oat::SinkState::UNDEFINED}; //!< SINK state
    std::atomic<size_t> source_read_count_ {0}; //!< Number SOURCE reads that have occured since last sink reset
    std::bitset<NUM_SLOTS> source_slots_;

    std::atomic<size_t> source_ref_count_ {0}; //!< Number of SOURCES sharing this node
    std::atomic<uint64_t> write_number_ {0}; //!< Number of writes to shmem that have been facilited by this node

    // Unfortunately, must manually maintain the number of rbx_'s to match NUM_SLOTS
    semaphore mutex_ {1}; //!< mutex governing exclusive acces to the read_barrier_
    semaphore rb0_ {0}, rb1_ {0}, rb2_ {0}, rb3_ {0}, rb4_ {0},
              rb5_ {0}, rb6_ {0}, rb7_ {0}, rb8_ {0}, rb9_ {0};
};

} // namespace oat

#endif	/* OAT_NODE_H */

