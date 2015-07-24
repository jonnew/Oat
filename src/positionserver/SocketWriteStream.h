//******************************************************************************
//* File:   RJSocketWriteStream.h
//* Author: Jon Newman <jpnewman snail mit dot edu>
//
//* Copyright (c) Jon Newman (jpnewman snail mit dot edu) 
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

#ifndef SOCKETWRITESTREAM_H
#define	SOCKETWRITESTREAM_H

#include <cstdio>
#include <boost/asio.hpp>

#include "../../lib/rapidjson/rapidjson.h"

RAPIDJSON_NAMESPACE_BEGIN

//! Wrapper of C network ouput stream using sendto().
/*!
    \note implements Stream concept
*/
template <typename S, typename E>
class SocketWriteStream {
public:
    typedef char Ch;    //!< Character type. Only support char.

    SocketWriteStream(S* socket, const E& endpoint, char* buffer, size_t bufferSize) : 
      socket_(socket)
    , endpoint_(endpoint)
    , buffer_(buffer)
    , bufferEnd_(buffer + bufferSize)
    , current_(buffer_) { 
        
        RAPIDJSON_ASSERT(socket_ != 0);
    }

    void Put(char c) { 
        if (current_ >= bufferEnd_)
            Flush();

        *current_++ = c;
    }

    void PutN(char c, size_t n) {
        size_t avail = static_cast<size_t>(bufferEnd_ - current_);
        while (n > avail) {
            std::memset(current_, c, avail);
            current_ += avail;
            Flush();
            n -= avail;
            avail = static_cast<size_t>(bufferEnd_ - current_);
        }

        if (n > 0) {
            std::memset(current_, c, n);
            current_ += n;
        }
    }

    void Flush() {
        if (current_ != buffer_) {
            // TODO: Blocks - should I use the async version?
            socket_->send_to(boost::asio::buffer(buffer_, static_cast<size_t>(current_ - buffer_)), endpoint_);
            current_ = buffer_;
        }
    }

    // Not implemented
    char Peek() const { RAPIDJSON_ASSERT(false); return 0; }
    char Take() { RAPIDJSON_ASSERT(false); return 0; }
    size_t Tell() const { RAPIDJSON_ASSERT(false); return 0; }
    char* PutBegin() { RAPIDJSON_ASSERT(false); return 0; }
    size_t PutEnd(char*) { RAPIDJSON_ASSERT(false); return 0; }
    
    // Prohibit copy constructor & assignment operator.
    SocketWriteStream(const SocketWriteStream&) = delete;
    SocketWriteStream& operator=(const SocketWriteStream&) = delete;

private:
    
    S* socket_;
    const E endpoint_;
    char* buffer_;
    char* bufferEnd_;
    char* current_;
};

//! TODO; SImplement specialized version of PutN() with memset() for better performance.
//template <typename S, typename E>
//inline void PutN(SocketWriteStream<S,E>& stream, char c, size_t n) {
//    stream.PutN(c, n);
//}

RAPIDJSON_NAMESPACE_END

#endif	/* RJSOCKETWRITESTREAM_H */

