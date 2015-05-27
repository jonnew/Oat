//******************************************************************************
//* Copyright (c) Jon Newman (jpnewman at mit snail edu) 
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

#include "BackgroundSubtractor.h"

#include <signal.h>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

volatile sig_atomic_t done = 0;
bool running = true;

void term(int) {
    done = 1;
}

void run(BackgroundSubtractor* background_subtractor, std::string source, std::string sink) {

    std::cout << "Background subtractor has begun listening to source \"" + source + "\".\n";
    std::cout << "Background subtractor has begun steaming to sink \"" + sink + "\".\n";

    while (!done) {

        background_subtractor->filterAndServe();
    }

    std::cout << "Background subtractor is exiting.\n";
}

void printUsage(po::options_description options) {
    std::cout << "Usage: backsub [OPTIONS]\n";
    std::cout << "   or: backsub SOURCE SINK\n"; // TODO: TYPE
    std::cout << "Perform background subtraction on images from SOURCE.\n";
    std::cout << "Publish background-subtracted images to SMServer<SharedCVMatHeader> SINK.\n";
    std::cout << options << "\n";
}

int main(int argc, char *argv[]) {

    signal(SIGINT, term);

    std::string source;
    std::string sink;

    try {

        po::options_description options("OPTIONS");
        options.add_options()
                ("help", "Produce help message.")
                ("version,v", "Print version information.")
                ;

        po::options_description hidden("HIDDEN OPTIONS");
        hidden.add_options()
                ("source", po::value<std::string>(&source),
                "The name of the SOURCE that supplies images on which to perform background subtraction."
                "The server must be of type SMServer<SharedCVMatHeader>\n")
                ("sink", po::value<std::string>(&sink),
                "The name of the SINK to which background subtracted images will be served.")
                ;

        po::positional_options_description positional_options;
        positional_options.add("source", 1);
        positional_options.add("sink", 2);

        po::options_description all_options("All options");
        all_options.add(options).add(hidden);

        po::variables_map variable_map;
        po::store(po::command_line_parser(argc, argv)
                .options(all_options)
                .positional(positional_options)
                .run(),
                variable_map);
        po::notify(variable_map);

        // Use the parsed options
        if (variable_map.count("help")) {
            printUsage(options);
            return 0;
        }

        if (variable_map.count("version")) {
            std::cout << "Simple-Tracker Background Subtractor version 1.0\n"; //TODO: Cmake managed versioning
            std::cout << "Written by Jonathan P. Newman in the MWL@MIT.\n";
            std::cout << "Licensed under the GPL3.0.\n";
            return 0;
        }

        if (!variable_map.count("source")) {
            printUsage(options);
            std::cout << "Error: a SOURCE must be specified. Exiting.\n";
            return -1;
        }

        if (!variable_map.count("sink")) {
            printUsage(options);
            std::cout << "Error: a SINK name must be specified. Exiting.\n";
            return -1;
        }


    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Exception of unknown type! " << std::endl;
    }

    BackgroundSubtractor background_subtractor(source, sink);

    // Two threads - one for user interaction, the other
    // for executing the processor
    boost::thread_group thread_group;
    thread_group.create_thread(boost::bind(&run, &background_subtractor, source, sink));
    sleep(1);

    // Start the user interface
    while (!done) {

        char user_input;
        std::cout << "Select an action:\n";
        //std::cout << " b: Set background image to current\n";
        std::cout << " x: Exit\n";
        std::cout << ">> ";

        std::cin >> user_input;

        switch (user_input) {
//            case 'b':
//            {
//                background_subtractor.setBackgroundImage();
//                break;
//            }
            case 'x':
            {
                done = true;
                break;
            }
            default:
                std::cout << "Invalid selection. Try again.\n";
                break;
        }
    }

    // Join processing and UI threads
    thread_group.join_all();

    // Exit
    return 0;
}


