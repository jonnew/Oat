//******************************************************************************
//* File:   Source.h
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

#ifndef OAT_SOURCE_H
#define	OAT_SOURCE_H

#include "ForwardsDecl.h"
#include "Node.h"
#include "SharedFrameHeader.h"

#include <exception>
#include <iostream>
#include <memory>
#include <string>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/thread/thread_time.hpp>

#include "../datatypes/Frame2.h"
#include "../utility/make_unique.h"

#include "../base/Globals.h"

namespace oat {

enum class SourceState {
    err_connect = -3,
    err_nodefull = -2,
    err_typemis = -1,
    virgin = 0,
    touched = 1,
    connected = 2,
};

template <typename T, typename ReturnT = T>
class SourceBase {
public:
    explicit SourceBase(const std::string &addr)
    : address(addr)
    , node_address_(addr + "_node")
    , obj_address_(addr + "_obj")
    {
        // Make sure we did not connect already
        if (state_ != SourceState::virgin) {
            throw std::runtime_error("A source can only connect a "
                                     "single time to a single node.");
        }

        // Create shmem segments
        node_shmem_ = bip::managed_shared_memory(
            bip::open_or_create, node_address_.c_str(), EXTRA + sizeof(Node));

        // Create or find Node which facilitates synchronized access to shmem
        node_ = node_shmem_.template find_or_construct<Node>(
            typeid(Node).name())();

        // Let the node know this source is attached and retrieve *this's index
        if (node_->acquireSlot(slot_index_) < 0) {
            state_ = SourceState::err_nodefull;
            throw std::runtime_error(
                "Node at " + address
                + "has reached capacity and this component cannot connect.");
        }

        // We have touched the node and must sychronize with its sink
        state_ = SourceState::touched;
    }

    ~SourceBase()
    {
        // If we have touched the node, or there was a node type mismatch, we
        // must
        // release our slot
        if (state_ >= SourceState::touched
            || state_ == SourceState::err_typemis)
            node_->releaseSlot(slot_index_);

        // If the client reference count is 0 and there is no sink
        // attached to the node, deallocate the shmem
        if ((node_ != nullptr && node_->source_ref_count() == 0)
            && node_->sink_state != Node::State::sink_bound) {

            bip::shared_memory_object::remove(node_address_.c_str());
            bip::shared_memory_object::remove(obj_address_.c_str());
        }
    }

    SourceState connect(void)
    {
        // Make sure we did not connect already
        if (state_ != SourceState::touched)
            throw std::runtime_error("A source can only connect() after it has "
                                     "touch()ed a node.");

        // Wait for the SINK to bind and construct the shared object
        if (node_->sink_state != Node::State::sink_bound) {

            // No throw because this can occur at quit
            if (wait() != Node::State::sink_bound)
                return SourceState::err_connect;

            // Self post since all loops start with wait() and we just
            // finished our wait(). This will make the first call to
            // wait() a 'freebie'
            node_->read_barrier(slot_index_).post();
            wait_complete_ = false;
        }

        // Find an existing shared object constructed by the SINK
        obj_shmem_
            = bip::managed_shared_memory(bip::open_only, obj_address_.c_str());
        std::pair<T *, std::size_t> temp = obj_shmem_.find<T>(typeid(T).name());
        sh_object_ = temp.first;

        // Only occurs when the name of the shared object does not match
        // typeid(T).name()
        if (sh_object_ == nullptr) {
            state_ = SourceState::err_typemis;
            throw std::runtime_error(
                "Type mismatch: Source<T> can only connect to Node<T>.");
        }

        state_ = SourceState::connected;
        return state_;
    }

    // Sychronization
    Node::State wait()
    {
        assert(!wait_complete_ && "wait() called when post() was required.");

        boost::system_time timeout = boost::get_system_time() + msec_t(10);

        // Only wait if there is a SOURCE attached to the node
        // Wait with timed wait with period check to prevent deadlocks
        while (!node_->read_barrier(slot_index_).timed_wait(timeout) && !quit) {

            // Loops checking if wait has been released
            timeout = boost::get_system_time() + msec_t(10);

            // If the sink has left the room, we should too
            if (node_->sink_state == Node::State::end)
                break;
        }

        wait_complete_ = true;

        return node_->sink_state;
    }

    void post()
    {
        assert(state_ == SourceState::connected
               && "source must be connected before calling post()");
        assert(wait_complete_
               && "post() called when wait() was required.");

        if (node_->notifySourceReadComplete(slot_index_))
            node_->write_barrier.post();

        wait_complete_ = false;
    }

    T *retrieve()
    {
        assert(state_ == SourceState::connected
               && "Source must be connected before shared object is retrieved.");

        return sh_object_;
    }

    T clone() const
    {
        assert(state_ == SourceState::connected
               && "Source must be connected before shared object is cloned.");

        return *sh_object_;
    }

    int pull(ReturnT **t)
    {
        assert(state_ == SourceState::connected
               && "Source must be connected before shared object is pulled.");

        auto rc = wait();
        if (rc == oat::Node::State::end)
            return 1;
        *t = sh_object_; // No copy; dereferencing (**t) _is_ *sh_object
        post();

        return 0;
    }

    int pull(std::unique_ptr<ReturnT> &clone)
    {
        assert(state_ == SourceState::connected
               && "Source must be connected before shared object is pulled.");

        auto rc = wait();
        if (rc == oat::Node::State::end)
            return 1;
        clone = oat::make_unique<T>(*sh_object_); // Copy construct.
        post();

        return 0;
    }

    int pull(ReturnT &clone)
    {
        assert(state_ == SourceState::connected
               && "Source must be connected before shared object is pulled.");

        auto rc = wait();
        if (rc == oat::Node::State::end)
            return 1;
        clone = *sh_object_; // Copy assign
        post();

        return 0;
    }

    uint64_t write_number() const
    {
        return (node_ == nullptr ? 0 : node_->write_number());
    }

    const std::string address;
protected:

    shmem_t node_shmem_, obj_shmem_;
    T *sh_object_{nullptr};
    Node *node_{nullptr};
    const std::string node_address_, obj_address_;
    size_t slot_index_{0};
    std::atomic<SourceState> state_{SourceState::virgin};
    bool wait_complete_{false};
};

// For types that do not use heap allocation
template<typename T>
using Source = SourceBase<T, T>;

// For Frame
using FrameSource = SourceBase<oat::SharedFrame, oat::Frame>;

// FrameSource pull specilizations
template <>
inline int FrameSource::pull(oat::Frame &clone)
{
    assert(state_ == SourceState::connected
           && "Source must be connected before shared object is pulled.");

    auto rc = wait();
    if (rc == oat::Node::State::end)
        return 1;
    clone = oat::copyFrame<oat::SharedFrame, oat::Frame>(*sh_object_); // Copy assign
    post();

    return 0;
}

}      /* namespace oat */
#endif /* OAT_SOURCE_H */
