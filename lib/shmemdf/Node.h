//******************************************************************************
//* File:   Node.h
//* Author: Jon Newman <jpnewman snail mit dot edu>
//*
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
#include <iostream>
#include <string>

#include <boost/interprocess/sync/interprocess_semaphore.hpp>

#include "ForwardsDecl.h"

// Extra bytes are used to hold managed shared memory overhead
// (name-object index, internal synchronization objects, internal
// variables...)
#define EXTRA 1024

namespace oat {

class Node {

    using Semaphore = bip::interprocess_semaphore;

public:
    enum class State {
        undefined = 0,
        sink_present,
        sink_bound,
        error,
        end,
    };

    /** 
     * @brief Number of sources that can attach 
     */
    static constexpr size_t num_slots{10};

    Node()
    {
        source_slots_.reset();
        source_read_required_.reset();
    }

    // Nodes are not copyable
    Node(const Node &) = delete;
    Node &operator=(const Node &) = delete;

    // Nodes are movable
    // TODO: These are default deleted since the copy ctor is deleted!
    Node(Node &&node) = default;
    Node &operator=(Node &&) = default;

    // SINK state
    std::atomic<State> sink_state{State::undefined}; //!< SINK state

    // SINK writes (~sample number)
    uint64_t write_number() const { return write_number_; }

    void notifySinkWriteComplete()
    {
        mutex_.wait();

        // Require one read from all connected sources
        source_read_required_ = source_slots_;

        // Tell each source connected to the node that it may read
        for (size_t i = 0; i < source_slots_.size(); i++)
            if (source_slots_[i])
                read_barrier(i).post();

        ++write_number_;

        mutex_.post();
    }

    // SOURCE read counting
    bool notifySourceReadComplete(size_t index)
    {
        mutex_.wait();

        source_read_required_[index] = false;
        bool reads_finished = source_read_required_.none();

        mutex_.post();

        return reads_finished;
    }


    int acquireSlot(size_t &index)
    {
        mutex_.wait();

        if (source_slots_.all()) {
            mutex_.post();
            return -1;
        }

        index = 0;
        while (source_slots_[index])
            ++index;

        source_slots_[index] = true;
        source_ref_count_ = source_slots_.count();

        mutex_.post();

        return 0;
    }

    int releaseSlot(size_t index)
    {
        if (index >= source_slots_.size())
            return -1;

        mutex_.wait();
        source_slots_[index] = false;
        source_ref_count_ = source_slots_.count();
        mutex_.post();

        return 0;
    }

    size_t source_ref_count(void) const { return source_ref_count_; }

    // Synchronization constructs
    // write _always_ occurs before read. By starting at 1, the writer is not
    // blocked by an initial wait. Readers to do not post to the write_barrier
    // until a write occurs.
    Semaphore write_barrier{1};

    // This method is required because an std::array of semaphores requires
    // each Semaphore to be copy-constructed to initialized the array.
    // Because of their nature, semaphores are NOT copy constructable, so
    // this approach does not work.
    Semaphore &read_barrier(size_t index)
    {
        if (!source_slots_[index])
            throw std::runtime_error("Requested index refers to a SOURCE "
                                     "that is not bound to this node.");

        switch (index) {
            case 0: return rb0_; break;
            case 1: return rb1_; break;
            case 2: return rb2_; break;
            case 3: return rb3_; break;
            case 4: return rb4_; break;
            case 6: return rb6_; break;
            case 7: return rb7_; break;
            case 8: return rb8_; break;
            case 9: return rb9_; break;
            default:
                throw std::runtime_error("Source index out of range.");
                break;
        }
    }

private:

    std::bitset<num_slots> source_slots_;
    std::bitset<num_slots> source_read_required_;

    size_t source_ref_count_{0}; //!< Number of SOURCES sharing this node
    uint64_t write_number_{0}; //!< Number of writes to shmem that have been facilited by this node

    // Unfortunately, must manually maintain the number of rbx_'s to match num_slots
    Semaphore mutex_{1}; //!< mutex governing exclusive acces to the read_barrier_
    Semaphore rb0_{0}, rb1_{0}, rb2_{0}, rb3_{0}, rb4_{0},
              rb5_{0}, rb6_{0}, rb7_{0}, rb8_{0}, rb9_{0};
};

}       /* namespace oat */
#endif	/* OAT_NODE_H */
