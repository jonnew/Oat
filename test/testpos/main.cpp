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

#include <unordered_map>
#include <time.h>
#include <signal.h>
#include <memory>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/program_options.hpp>
#include <boost/any.hpp>

#include "TestPosition.h"
#include "RandomAccel2D.h"

namespace po = boost::program_options;

volatile sig_atomic_t done = 0;
bool running = true;
struct timespec delay = {0};


void term(int) {
    done = 1;
}

void printUsage(po::options_description options) {
    std::cout << "Usage: testpos [OPTIONS]\n"
              << "   or: testpos TYPE SINK [CONFIGURATION]\n"
              << "Publish test positions to a SMSserver<Position> SINK.\n\n"
              << "TYPE\n"
              << "  'rand2D': Randomly accelerating 2D Position\n"
              << "  'rand3D': Randomly accelerating 3D Position (W.I.P.)\n\n"
              << options << "\n";
}

// Processing thread

void run(TestPosition<datatypes::Position2D>* test_position) {

    while (!done) {
        test_position->simulateAndServePosition();
        nanosleep(&delay, (struct timespec *)NULL);
    }
}

// IO thread

int main(int argc, char *argv[]) {

    delay.tv_sec = 0;
    delay.tv_nsec = (int)(DT * 1.0e9);

    // Base options
    po::options_description visible_options("VISIBLE OPTIONS");

    std::string sink;
    std::string type;
    std::string config_file;
    std::string config_key;
    bool config_used = false;
    
    
    std::unordered_map<std::string, char> type_hash;
    type_hash["rand2D"] = 'a';
    //TODO: type_hash["rand3D"] = 'b';  


    try {

        po::options_description options("OPTIONS");
        options.add_options()
                ("help", "Produce help message.")
                ("version,v", "Print version information.")
                ;

        po::options_description config("CONFIGURATION");
        config.add_options()
                ("config-file,c", po::value<std::string>(&config_file), "Configuration file.")
                ("config-key,k", po::value<std::string>(&config_key), "Configuration key.")
                ;

        po::options_description hidden("HIDDEN OPTIONS");
        hidden.add_options()
                ("type", po::value<std::string>(&type), 
                "Type of test position to serve.")
                ("sink", po::value<std::string>(&sink),
                "The name of the SINK to which position background subtracted images will be served.")
                ;

        po::positional_options_description positional_options;
        positional_options.add("type", 1);
        positional_options.add("sink", 1);

        po::options_description visible_options("VISIBLE OPTIONS");
        visible_options.add(options).add(config);

        po::options_description all_options("ALL OPTIONS");
        all_options.add(options).add(config).add(hidden);

        po::variables_map variable_map;
        po::store(po::command_line_parser(argc, argv)
                .options(all_options)
                .positional(positional_options)
                .run(),
                variable_map);
        po::notify(variable_map);

        // Use the parsed options
        if (variable_map.count("help")) {
            printUsage(visible_options);
            return 0;
        }

        if (variable_map.count("version")) {
            std::cout << "Simple-Tracker Test Position Server version 1.0\n" 
                      << "Written by Jonathan P. Newman in the MWL@MIT.\n"
                      << "Licensed under the GPL3.0.\n";
            return 0;
        }
        
        if (!variable_map.count("type")) {
            printUsage(visible_options);
            std::cout << "Error: a TYPE name must be specified. Exiting.\n";
            return -1;
        }

        if (!variable_map.count("sink")) {
            printUsage(visible_options);
            std::cout << "Error: a SINK name must be specified. Exiting.\n";
            return -1;
        }

        if ((variable_map.count("config-file") && !variable_map.count("config-key")) ||
                (!variable_map.count("config-file") && variable_map.count("config-key"))) {
            printUsage(visible_options);
            std::cout << "Error: config file must be supplied with a corresponding config-key. Exiting.\n";
            return -1;
        } else if (variable_map.count("config-file")) {
            config_used = true;
        }

    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Exception of unknown type!" << std::endl;
    }

    // Create the specified TYPE of detector
    // TODO: Right now I need to use datatypes::Position2D as a template parameter
    // because otherwise this is no longer a valid base class for RandomAccel2D whose
    // base class is indeed TestPosition<datatypes::Position2D>
    TestPosition<datatypes::Position2D>* test_position;
    
    switch (type_hash[type]) {
        case 'a':
        {
            test_position = new RandomAccel2D(sink);
            break;
        }
        default:
        {
            printUsage(visible_options);
            std::cout << "Error: invalid TYPE specified. Exiting.\n";
            return -1;
        }
    }
    
    //if (config_used)
    //test_position->configure(config_file, config_key);
    
    std::cout << "Test Position has begun publishing to sink \"" + sink + "\".\n\n"
              << "COMMANDS:\n"
              << "  p: Pause/unpause.\n"
              << "  x: Exit.\n\n";

    // Two threads - one for user interaction, the other
    // for executing the processor
    boost::thread_group thread_group;
    thread_group.create_thread(boost::bind(&run, test_position));
    sleep(1);

    while (!done) {
        
        char user_input;
        std::cin >> user_input;

        switch (user_input) {

            case 'p':
            {
                running = !running;
                break;
            }
            case 'x':
            {
                done = true;
                test_position->stop(); 
                break;
            }
            default:
                std::cout << "Invalid command. Try again.\n";
                break;
        }
    }

    // Exit gracefully and ensure all shared resources are cleaned up!
    thread_group.join_all();
    
    // Free heap memory allocated to test_position 
    delete test_position;

    std::cout << "Test Position is exiting.\n";

    // Exit
    return 0;
}


