//******************************************************************************
//* File:   Sink_test.cpp
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

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <string>

#include "../../lib/datatypes/Pixel.h"
#include "../../lib/datatypes/Frame2.h"
#include "../../lib/shmemdf/Sink2.h"

const std::string node_addr = "test";

// Global via extern in Globals.h
namespace oat {
volatile sig_atomic_t quit = 0;
}

SCENARIO("Sinks can bind a single Node.", "[Sink]")
{
    GIVEN("Two Sink<int>'s")
    {
        WHEN("sink1 creates a named node.")
        {
            oat::Sink<int> sink1(node_addr);

            THEN("An attempt create a sink with the same named node shall throw.")
            {
                REQUIRE_THROWS([&]() { oat::Sink<int> sink2(node_addr); }());
            }
        }
    }
}

SCENARIO("Sinks must bind() before waiting or posting.", "[Sink]")
{
    GIVEN("A single Sink<int>")
    {
        oat::Sink<int> sink(node_addr);

        // NB: Changed to assert
        //WHEN("When the sink calls wait() before binding a segment")
        //{
        //    THEN("The the sink shall throw") { REQUIRE_THROWS(sink.wait()); }
        //}

        // NB: Changed to assert
        //WHEN("When the sink calls post() before binding a segment")
        //{
        //    THEN("The the sink shall throw") { REQUIRE_THROWS(sink.post()); }
        //}

        WHEN("When the sink calls wait() after binding a segment")
        {
            sink.bind();

            THEN("The the sink shall not throw")
            {
                REQUIRE_NOTHROW(sink.wait());
            }
        }

        // NB: Changed to assert
        //WHEN("When the sink calls post() after binding a segment "
        //     "before calling wait()")
        //{
        //    sink.bind();

        //    THEN("The the sink shall throw") { REQUIRE_THROWS(sink.post()); }
        //}

        WHEN("When the sink calls post() after binding a segment "
             "after calling wait()")
        {
            sink.bind();
            sink.wait();

            THEN("The the sink shall not throw")
            {
                REQUIRE_NOTHROW(sink.post());
            }
        }
    }
}

SCENARIO("Sinks cannot bind() to the same node more than once.", "[Sink]")
{
    oat::Sink<int> sink(node_addr);

    INFO("The sink binds a node");
    REQUIRE_NOTHROW(sink.bind());
    REQUIRE_THROWS(sink.bind());
}

SCENARIO("Bound sinks can retrieve shared objects to mutate them.", "[Sink]")
{
    GIVEN("A single Sink<int> and a shared *int=0")
    {
        int *shared = static_cast<int *>(0);
        oat::Sink<int> sink(node_addr);

        // NB: Changed to assert
        //WHEN("When the sink calls retrieve() before binding a segment")
        //{

        //    THEN("The the sink shall throw")
        //    {
        //        REQUIRE_THROWS(shared = sink.retrieve());
        //    }
        //}

        WHEN("When the sink calls retrieve() after binding a segment")
        {

            sink.bind();

            THEN("The the sink returns a pointer to mutate the shared integer")
            {

                INFO("Start with shared int uninitialized")
                CAPTURE(shared);

                REQUIRE_NOTHROW(shared = sink.retrieve());

                INFO("Set shared int to 1")
                *shared = 1;
                CAPTURE(*shared);

                REQUIRE(*shared == *sink.retrieve());
            }
        }
    }
}

SCENARIO("Sink<SharedFrameHeader> must bind() before waiting, posting, or "
         "allocating.",
         "[Sink, SharedFrameHeader]")
{
    GIVEN("A single Sink<SharedFrameHeader>")
    {
        oat::Sink<oat::SharedFrame, oat::SharedFrameAllocator> sink(node_addr);
        oat::SharedFrame * frame;
        size_t cols{100};
        size_t rows{100};
        oat::Pixel::Color color{oat::Pixel::Color::bgr};

        // NB: Changed to assert
        //WHEN("When the sink calls wait() before binding a segment")
        //{
        //    THEN("The the sink shall throw") { REQUIRE_THROWS(sink.wait()); }
        //}

        // NB: Changed to assert
        //WHEN("When the sink calls post() before binding a segment")
        //{
        //    THEN("The the sink shall throw") { REQUIRE_THROWS(sink.post()); }
        //}

        // NB: Changed to assert
        //WHEN("When the sink calls retrieve() before binding a segment")
        //{
        //    THEN("The the sink shall throw")
        //    {
        //        REQUIRE_THROWS([&]() { frame = sink.retrieve(); }());
        //    }
        //}

        WHEN("When the sink calls retrieve() after binding and reserving a segment")
        {
            THEN("The the sink shall not throw")
            {
                oat::Frame f(10, rows, cols, color);
                sink.reserve(f.bytes());
                sink.bind(10, rows, cols, color);
                REQUIRE_NOTHROW([&]() { frame = sink.retrieve(); }());
            }
        }
    }
}
