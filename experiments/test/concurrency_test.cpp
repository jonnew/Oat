//******************************************************************************
//* File:   concurrency_test.cpp
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

#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main()
#include <catch.hpp>
//#include "/home/jon/Public/Oat/debug/catch/src/catch/include/catch.hpp"

#include <thread> // Simulate different processes

#include "../lib/Sink.h"
#include "../lib/Source.h"

// Test outline
//
//- Given a sink in the critical section
//    - It should not be possible for a source to enter
//
//- Given a source in the critical section
//    - It should not be possible for a sink to enter
//    - It should be possible for other sources to enter
//
//- Given a waiting sink
//    - A source should have to post for the sink to proceed
//
//- Given a waiting source
//    - A sink should have to post for the sink to proceed
//
//- Given a waiting source
//    - When a second source joins, both sources should proceed when a sink posts
//
//- Given two waiting sources
//    - When the first source leaves, the second source should proceed when a sink posts
