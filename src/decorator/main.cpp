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

#include "Decorator.h"

#include <string>
#include <signal.h>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

volatile sig_atomic_t done = 0;

void run(Decorator* decorator) {

    while (!done) {
        decorator->decorateAndServeImage();
    }
}

void printUsage(po::options_description options) {
    std::cout << "Usage: decorate [OPTIONS]\n";
    std::cout << "   or: decorate POSITION_SOURCES IMAGE_SOURCE IMAGE_SINK\n";
    std::cout << "Decorate the image provided by IMAGE_SOURCE using object position information from POSITION_SOURCES.\n";
    std::cout << "Publish decorated image to IMAGE_SINK.\n";
    std::cout << options << "\n";
}

int main(int argc, char *argv[]) {

    // The image source to which the viewer will be attached
    std::vector<std::string> sources;
    std::string image_source;
    std::string sink;

    try {

        po::options_description options("OPTIONS");
        options.add_options()
                ("help", "Produce help message.")
                ("version,v", "Print version information.")
                ;

        po::options_description hidden("HIDDEN OPTIONS");
        hidden.add_options()
                ("positionsources", po::value< std::vector<std::string> >(),
                "The name of the server(s) that supply object position information."
                "The server(s) must be of type SMServer<Position>\n")
                ("imagesource", po::value<std::string>(&image_source),
                "The name of the server that supplies images to decorate."
                "The server must be of type SMServer<SharedCVMatHeader>\n")
                ("sink", po::value<std::string>(&sink),
                "The name of the sink to which decorated images will be published."
                "The server must be of type SMServer<SharedCVMatHeader>\n")
                ;

        po::positional_options_description positional_options;
        
        // If not overridden by explicit --imagesource or --sink, 
        // last positional arguments are imagesource and sink in that order
        positional_options.add("positionsources", -1); 

        po::options_description all_options("ALL OPTIONS");
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
            std::cout << "Simple-Tracker Decorator, version 1.0\n"; //TODO: Cmake managed versioning
            std::cout << "Written by Jonathan P. Newman in the MWL@MIT.\n";
            std::cout << "Licensed under the GPL3.0.\n";
            return 0;
        }

        if (!variable_map.count("positionsources")) {
            printUsage(options);
            std::cout << "Error: at least a single POSITION_SOURCE must be specified. Exiting.\n";
            return -1;
        }
        
        // May contain imagesource and sink information!
        sources = variable_map["positionsources"].as< std::vector<std::string> >(); 
        
        if (!variable_map.count("imagesource") && !variable_map.count("sink")) {
            sink = sources.back();
            sources.pop_back(); 
            
            image_source = sources.back();
            sources.pop_back(); 
        }
        else if (variable_map.count("imagesource") && !variable_map.count("sink")) {
            sink = sources.back();
            sources.pop_back(); 
        }
        else if (!variable_map.count("imagesource") && variable_map.count("sink")) {
            image_source = sources.back();
            sources.pop_back();
        }

    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Exception of unknown type! " << std::endl;
    }
    
    std::cout << "Decorator named \"" + sink + "\" has started.\n";
    std::cout << "COMMANDS:\n";
    std::cout << "  x: Exit.\n";

    // Make the decorator
    Decorator decorator(sources, image_source, sink);

    // Two threads - one for user interaction, the other
    // for processing
    boost::thread_group thread_group;
    thread_group.create_thread(boost::bind(&run, &decorator));
    sleep(1);

    // Start the user interface
    while (!done) {

        char user_input;
        std::cin >> user_input;

        switch (user_input) {
            case 'x':
            {
                done = true;
                decorator.stop();
                break;
            }
            default:
                std::cout << "Invalid selection. Try again.\n";
                break;
        }
    }

    // Join processing and UI threads
    thread_group.join_all();
    
    std::cout << "Decorator named \"" + decorator.get_name() + "\" is exiting." << std::endl;

    // Exit
    return 0;
}
