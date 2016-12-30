//******************************************************************************
//* File:   Helpers_test.cpp
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

#include "../../lib/shmemdf/Helpers.h"

// Global via extern in Globals.h
namespace oat { volatile sig_atomic_t quit = 0; }

SCENARIO ("Sample period comparisons with epsilon tolerance.", "[Helpers]") {

    GIVEN ("An epsilon tolerance for sample periods.") {

        double min_rate;
        const double epsilon = 1e-6;

        WHEN ("The difference between the max and min sample periods is greater than epsilon.") {

            std::vector<double> all_ts {10, 10 + epsilon, 10 - epsilon};

            THEN ("The sample periods are inconsistent.") {

                bool ts_consistent = oat::checkSamplePeriods(all_ts, min_rate, epsilon);
                REQUIRE (!ts_consistent);
                REQUIRE (min_rate == Approx(1.0 / (10 - epsilon)));
            }
        }

        WHEN ("The difference between the max and min sample periods is less than or equal to epsilon.") {

            double div = 2.01;
            std::vector<double> all_ts {10, 10 + epsilon/div, 10 - epsilon/div};

            THEN ("The sample periods are consistent.") {

                bool ts_consistent = oat::checkSamplePeriods(all_ts, min_rate, epsilon);
                REQUIRE (ts_consistent);
                REQUIRE (min_rate == Approx(1.0 / (10 - epsilon/div)));
            }
        }
    }
}
