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
#include <sstream>
#include <string>
#include <thread>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/thread/thread_time.hpp>

#include "../datatypes/Frame.h"
#include "../base/Globals.h"

namespace oat {

enum class SourceState : std::int16_t
{
    ERR_CONNECT     = -3,
    ERR_NODEFULL    = -2,
    ERR_TYPEMIS     = -1,
    VIRGIN          = 0,
    TOUCHED         = 1,
    CONNECTED       = 2,
};

template <typename T>
class SourceBase {
public:
    SourceBase();
    virtual ~SourceBase();

    // Node connection
    void touch(const std::string &address);
    virtual SourceState connect(void);

    // Sychronization
    NodeState wait();
    void post();

    uint64_t write_number() const
    {
        return (node_ == nullptr ? 0 : node_->write_number());
    }

protected:

    shmem_t node_shmem_, obj_shmem_;
    T * sh_object_ {nullptr};
    Node * node_ {nullptr};
    std::string address_, node_address_, obj_address_;
    size_t slot_index_ {0};
    std::atomic<SourceState> state_ {SourceState::VIRGIN};
    bool touched_ {false};
    bool connected_ {false};
    bool did_wait_need_post_ {false};
};

template <typename T>
inline SourceBase<T>::SourceBase()
{
    // Nothing
}

template <typename T>
inline SourceBase<T>::~SourceBase()
{
    // If we have touched the node, or there was a node type mismatch, we must
    // release our slot
    if (state_ >= SourceState::TOUCHED || state_ == SourceState::ERR_TYPEMIS)
        node_->releaseSlot(slot_index_);

    // If the client reference count is 0 and there is no server
    // attached to the node, deallocate the shmem
    if ( (node_ != nullptr && node_-> source_ref_count() == 0) &&
        node_->sink_state() != NodeState::SINK_BOUND) {

        bool shmem_freed = false;
        shmem_freed |= bip::shared_memory_object::remove(node_address_.c_str());
        shmem_freed |= bip::shared_memory_object::remove(obj_address_.c_str());

#ifndef NDEBUG
        if (shmem_freed)
            std::cout << "Shared memory at \'" + address_ + "\' was deallocated.\n";
#endif
    }
}

template <typename T>
inline void SourceBase<T>::touch(const std::string &address)
{
    // Make sure we did not connect already
    if (state_ != SourceState::VIRGIN)
        throw std::runtime_error("A source can only connect a "
                                 "single time to a single node.");

    // Addresses for this block of shared memory
    address_ = address;
    node_address_ = address + "_node";
    obj_address_ = address + "_obj";

    // Define shared memory
    // Extra 1024 bytes are used to hold managed shared mem helper objects
    // (name-object index, internal synchronization objects, internal
    // variables...)
    node_shmem_ = bip::managed_shared_memory(
            bip::open_or_create,
            node_address_.c_str(),
            1024 + sizeof(Node));

    // Facilitates synchronized access to shmem
    node_ = node_shmem_.find_or_construct<Node>(typeid(Node).name())();

    // Let the node know this source is attached and retrieve *this's index
    if (node_->acquireSlot(slot_index_) < 0) {
        state_ = SourceState::ERR_NODEFULL;
        return;
    }

    // We have touched the node and must sychronize with its sink
    state_ = SourceState::TOUCHED;
}

template <typename T>
inline SourceState SourceBase<T>::connect()
{
    // Make sure we did not connect already
    if (state_ != SourceState::TOUCHED)
        throw std::runtime_error("A source can only connect() after it has "
                                 "touch()ed a node.");

    // Wait for the SINK to bind and construct the shared object
    if (node_->sink_state() != NodeState::SINK_BOUND) {

        if (wait() != NodeState::SINK_BOUND)
            return SourceState::ERR_CONNECT; // No throw because this can occur
                                             // at quit

        // Self post since all loops start with wait() and we just
        // finished our wait(). This will make the first call to
        // wait() a 'freebie'
        node_->read_barrier(slot_index_).post();
        did_wait_need_post_ = false;
    }

    // Find an existing shared object constructed by the SINK
    obj_shmem_ =
            bip::managed_shared_memory(bip::open_only, obj_address_.c_str());
    std::pair<T *,std::size_t> temp = obj_shmem_.find<T>(typeid(T).name());
    sh_object_ = temp.first;

    // Only occurs when the name of the shared object does not match typeid(T).name()
    if (sh_object_ == nullptr) {
        state_ = SourceState::ERR_TYPEMIS;
        throw std::runtime_error("Type mismatch: Source<T> can only connect to Node<T>.");
    }

    state_ = SourceState::CONNECTED;
    return SourceState::CONNECTED;
}

template <typename T>
inline NodeState SourceBase<T>::wait()
{
#ifndef NDEBUG
    // Don't use Asserts because it does not clean shmem
    if(state_ < SourceState::TOUCHED)
        throw std::runtime_error("Source must have touched node before calling wait()");
    if (did_wait_need_post_)
        throw std::runtime_error("wait() called when post() was required.");
#endif

    boost::system_time timeout = boost::get_system_time() + msec_t(10);

    // Only wait if there is a SOURCE attached to the node
    // Wait with timed wait with period check to prevent deadlocks
    while (!node_->read_barrier(slot_index_).timed_wait(timeout) && !quit) {

        // Loops checking if wait has been released
        timeout = boost::get_system_time() + msec_t(10);

        // If the sink has left the room, we should too
        if (node_->sink_state() == NodeState::END)
            break;
    }

    did_wait_need_post_ = true;

    return node_->sink_state();
}

template <typename T>
inline void SourceBase<T>::post()
{
#ifndef NDEBUG
    // Don't use Asserts because it does not clean shmem
    if(state_ < SourceState::CONNECTED)
        throw std::runtime_error("source must be connected before calling post()");
    if (!did_wait_need_post_)
        throw std::runtime_error("post() called when wait() was required.");
#endif

    if (node_->notifySourceReadComplete(slot_index_))
        node_->write_barrier.post();

    did_wait_need_post_ = false;
}

/* SPECIALIZATIONS */

// 0. General Case

template <typename T>
class Source : public SourceBase<T> {

    using SourceBase<T>::sh_object_;
    using SourceBase<T>::connected_;
    using SourceBase<T>::state_;

public:
    T *retrieve() const;
    T clone() const;
};

template <typename T>
inline T *Source<T>::retrieve() const
{
#ifndef NDEBUG
    // Don't use Asserts because it does not clean shmem
    if(state_ < SourceState::CONNECTED)
        throw (std::runtime_error("Source must be connected before shared object is retrieved."));
#endif

    return sh_object_;
}

template <typename T>
inline T Source<T>::clone() const
{
#ifndef NDEBUG
    // Don't use Asserts because it does not clean shmem
    if(state_ < SourceState::CONNECTED)
        throw (std::runtime_error("Source must be connected before shared object is cloned."));
#endif

    return *sh_object_;
}

// 1. SharedFrameHeader

template <>
class Source<Frame> : public SourceBase<SharedFrameHeader> {
public:

    using FrameParams = oat::FrameParams;

    // TODO: This info is sitting inside SharedFrameHeader. Why am I creating a
    // new class here? This should be part of SharedFrameHeader so I can just
    // copy it out of there.
    SourceState connect() override;
    SourceState connect(const oat::PixelColor col);

    const oat::Frame * retrieve() const { return &frame_; }
    oat::Frame clone() const { return frame_.clone(); }
    void copyTo(oat::Frame &frame) const { frame_.copyTo(frame); };
    FrameParams parameters() const { return parameters_; }

private :

    // Shared frame
    oat::Frame frame_;
    FrameParams parameters_;
};

inline SourceState Source<Frame>::connect(const oat::PixelColor color)
{
    auto rc = connect();

    // Check frame pixel type if required
    if (color != PIX_ANY && frame_.color() != color) {
        throw std::runtime_error("Component requires frame source "
                                 "with pixels of type "
                                 + oat::color_str(color)
                                 + ". Maybe use oat-framefilt col?");
    }
    
    return rc;
}

inline SourceState Source<Frame>::connect()
{
    // Make sure we did not connect already
    if (state_ != SourceState::TOUCHED)
        throw std::runtime_error("A source can only connect() after it has "
                                 "touch()ed a node.");

    // Wait for the SINK to bind the node and provide matrix
    // header info.
    if (node_->sink_state() != NodeState::SINK_BOUND) {

        if (wait() != NodeState::SINK_BOUND)
            return SourceState::ERR_CONNECT; // No throw because this can occur
                                             // at quit

        // Self post since all loops start with wait() and we just
        // finished our wait(). This will make the first call to
        // wait() a 'freebie'
        node_->read_barrier(slot_index_).post();
        did_wait_need_post_ = false;
    }

    // Find an existing shared object constructed by the SINK
    obj_shmem_ =
            bip::managed_shared_memory(bip::open_only, obj_address_.c_str());
    std::pair<SharedFrameHeader *, size_t> temp =
            obj_shmem_.find<SharedFrameHeader>(typeid(SharedFrameHeader).name());
    sh_object_ = temp.first;

    // Only occurs when the name of the shared object does not match typeid(T).name()
    if (sh_object_ == nullptr) {
        state_ = SourceState::ERR_TYPEMIS;
        throw std::runtime_error("Type mismatch: Source<T> can only connect to Node<T>.");
    }

    // Generate frame header using info in shmem segment
    auto p = sh_object_->params();
    frame_ = oat::Frame(p.rows,
                        p.cols,
                        p.type,
                        p.color,
                        obj_shmem_.get_address_from_handle(sh_object_->data()),
                        obj_shmem_.get_address_from_handle(sh_object_->sample()));

    // Save parameters to construct cv::Mats with
    parameters_.cols = p.cols;
    parameters_.rows = p.rows;
    parameters_.type = p.type;
    parameters_.color = p.color;
    parameters_.bytes = frame_.total() * frame_.elemSize();

    state_ = SourceState::CONNECTED;
    return SourceState::CONNECTED;
}

}      /* namespace oat */
#endif /* OAT_SOURCE_H */
