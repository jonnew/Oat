//******************************************************************************
//* File:   Sink.h
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

#ifndef OAT_Sink_H
#define	OAT_Sink_H

#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/thread/thread_time.hpp>

#include "../base/Globals.h"
#include "../datatypes/Color.h"
#include "../datatypes/Frame2.h"
#include "../datatypes/Token.h"
#include "../utility/make_unique.h"

#include "ForwardsDecl.h"
#include "Node.h"

namespace oat {

template <typename T, typename PushT, typename Allocator>
class SinkBase {
public:
    explicit SinkBase(const std::string &address)
    : address_(address)
    , node_address_(address + "_node")
    , obj_address_(address + "_obj")
    {
        // Create shmem segments
        node_shmem_ = bip::managed_shared_memory(
            bip::open_or_create, node_address_.c_str(), EXTRA + sizeof(Node));
        obj_shmem_ = bip::managed_shared_memory(
            bip::create_only, obj_address_.c_str(), EXTRA + sizeof(T));

        // Create or find Node which facilitates synchronized access to shmem
        node_ = node_shmem_.template find_or_construct<Node>(
            typeid(Node).name())();

        if (node_->sink_state != Node::State::undefined) {
            throw std::runtime_error("Requested Sink address, '" + address_
                                     + "', is not available.");
        }

        node_->sink_state = Node::State::sink_present;
    }

    ~SinkBase()
    {
        node_->sink_state = Node::State::end;

        // If the client ref count is 0, memory can be deallocated
        if (node_->source_ref_count() == 0) {

            auto rc = bip::shared_memory_object::remove(node_address_.c_str());
            assert(rc && "Node shmem not deallocated when it should have been.");

            rc = bip::shared_memory_object::remove(obj_address_.c_str());
            assert(rc && "Shared object shmem not deallocated when it should have been.");
        }
    }

    // Sinks cannot be copied
    SinkBase&operator=(const SinkBase &) = delete;
    SinkBase(const SinkBase& orig) = delete;

    // TODO: remove enable_if from return type and put in template parameters.
    // It is ugly as return type

    template <typename A = Allocator>
    typename std::enable_if<!std::is_integral<A>::value, A>::type
    reserve(size_t bytes)
    {
        // Make sure there is not another Sink using this shmem
        if (node_->sink_state != Node::State::sink_present) {
            throw std::runtime_error("Requested Sink address, '" + address_
                                     + "', is not available.");
        } else {

            // Node is in undefined state, OK to destroy object memory and
            // resize
            bip::shared_memory_object::remove(obj_address_.c_str());
            obj_shmem_ = bip::managed_shared_memory(bip::create_only,
                                                    obj_address_.c_str(),
                                                    EXTRA + sizeof(T) + bytes);
            alloc_ = oat::make_unique<Allocator>(obj_shmem_.get_segment_manager());
        }

        return 0; // TODO: correct return type
    }

    template <typename... Targs, typename A = Allocator>
    typename std::enable_if<!std::is_integral<A>::value, A>::type
    reserve(size_t bytes, void **data)
    {
        // Make sure there is not another Sink using this shmem
        if (node_->sink_state != Node::State::sink_present) {
            throw std::runtime_error("Requested Sink address, '" + address_
                                     + "', is not available.");
        } else {

            // Node is in undefined state, OK to destroy object memory and
            // resize
            bip::shared_memory_object::remove(obj_address_.c_str());
            obj_shmem_ = bip::managed_shared_memory(bip::create_only,
                                                    obj_address_.c_str(),
                                                    EXTRA + sizeof(T) + bytes);
            *data = obj_shmem_.allocate(bytes); 
            data_handle_ = obj_shmem_.get_handle_from_address(*data); 
        }

        return 0; // TODO: correct return type
    }

    // Bind to Node (static specialization for custom alloctor)
    template <typename... Targs, typename A = Allocator>
    typename std::enable_if<!std::is_integral<A>::value, A>::type
    bind(Targs... args)
    {
        if (node_->sink_state == Node::State::sink_bound) {
            throw std::runtime_error("A sink can only bind a "
                                     "single time to a single node.");

        } else {

            // Find an existing shared object or construct one
            sh_object_ = obj_shmem_.template find_or_construct<T>(
                typeid(T).name())(args..., *alloc_);
            node_->sink_state = Node::State::sink_bound;
        }

        return 0; // TODO: correct return type
    }

    // Bind to Node (static specialization for void allocator)
    template <typename... Targs, typename A = Allocator>
    typename std::enable_if<std::is_integral<A>::value, A>::type
    bind(Targs... args)
    {
        if (node_->sink_state == Node::State::sink_bound) {
            throw std::runtime_error("A sink can only bind a "
                                     "single time to a single node.");
        } else {

            // Find an existing shared object or construct one
            sh_object_ = obj_shmem_.template find_or_construct<T>(
                typeid(T).name())(args...);
            node_->sink_state = Node::State::sink_bound;
        }

        return 0; // TODO: correct return type
    }

    T *retrieve()
    {
        assert (node_->sink_state == Node::State::sink_bound
               && "Sink must be bound before shared object is retrieved.");
        return sh_object_;
    }

    void wait()
    {
        assert(node_->sink_state == Node::State::sink_bound
               && "Sink must be bound before calling post()");
        assert(!wait_complete_ && "wait() called when post() was required.");

        boost::system_time timeout = boost::get_system_time() + msec_t(10);

        // Only wait if there is a SOURCE attached to the node
        // Wait with timed wait with period check to prevent deadlocks
        while (node_->source_ref_count() > 0
               && !node_->write_barrier.timed_wait(timeout)
               && !quit) {
            // Loops checking if wait has been released
            timeout = boost::get_system_time() + msec_t(10);
        }

        wait_complete_ = true;
    }

    void post()
    {
        assert(node_->sink_state == Node::State::sink_bound
               && "Sink must be bound before calling post()");
        assert(wait_complete_
               && "post() called when wait() was required.");

        // Increment the number times this node has facilitated a shmem write
        node_->notifySinkWriteComplete();
        wait_complete_ = false;

#ifndef NDEBUG
        // Flush to keep things in order
        std::cout << address_
                  << " completed write number: " << node_->write_number() - 1
                  << std::endl;
#endif
    }

    void push(PushT &&t) 
    {
        assert (node_->sink_state == Node::State::sink_bound
               && "Sink must be bound before calling push()");

        wait();
        *sh_object_ = std::move(t);
        post();
    }

private:
    std::string address_;
    shmem_t node_shmem_, obj_shmem_;
    Node *node_{nullptr};
    T *sh_object_{nullptr};
    std::string node_address_, obj_address_;
    bool wait_complete_{false};
    bip::managed_shared_memory::handle_t data_handle_;
    std::unique_ptr<Allocator> alloc_{nullptr};
};

// Tokens without heap usage
template <typename T>
using Sink = SinkBase<T, T, int>;

// Frames
using FrameSink
    = SinkBase<oat::SharedFrame, oat::Frame, oat::SharedFrameAllocator>;

// push() specializations
template <>
inline void FrameSink::push(oat::Frame &&t)
{
    assert(node_->sink_state == Node::State::sink_bound
           && "Sink must be bound before calling push()");

    wait();
    sh_object_->setTime(t.tick(), t.template time<Token::Seconds>());
    std::move(t.storage.begin(),
              t.storage.end(),
              std::back_inserter(sh_object_->storage));
    post();
}

}      /* namespace oat */
#endif /* OAT_Sink_H */
