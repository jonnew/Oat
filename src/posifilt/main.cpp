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
#include <string>
#include <signal.h>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/program_options.hpp>

#include "KalmanFilter2D.h"

namespace po = boost::program_options;

volatile sig_atomic_t done = 0;
bool running = true;

void printUsage(po::options_description options) {
    std::cout << "Usage: posifilt [OPTIONS]\n";
    std::cout << "   or: posifilt TYPE SOURCE SINK [CONFIGURATION]\n";
    std::cout << "Perform TYPE position filter on a position stream published by a SMServer<Position> SOURCE.\n";
    std::cout << "Publish filtered object position to a SMSserver<Position> SINK.\n\n";
    std::cout << "TYPE\n";
    std::cout << "  \'kalman\': Kalman filter\n\n";
    std::cout << options << "\n";
}

void run(PositionFilter* positionFilter) {

    while (!done) {
        positionFilter->filterPositionAndServe();
    }
}

int main(int argc, char *argv[]) {

    // The image source to which the viewer will be attached
    std::string source;
    std::string sink;
    std::string type;
    std::string config_file;
    std::string config_key;
    bool config_used = false;
    po::options_description visible_options("VISIBLE OPTIONS");

    std::unordered_map<std::string, char> type_hash;
    type_hash["kalman"] = 'a';
    //TODO: there should be a runtime decision, based on the types of positions
    //from the source, whether or not to use the 2D or 3D filter. Or, these types
    //of filter should be able to handle 2D or 3D positions

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
                ("type", po::value<std::string>(&type), "Filter TYPE.")
                ("positionsource", po::value<std::string>(&source),
                "The name of the server that supplies object position information."
                "The server must be of type SMServer<Position>\n")
                ("sink", po::value<std::string>(&sink),
                "The name of the sink to which filtered positions will be published."
                "The server must be of type SMServer<Position>\n")
                ;

        po::positional_options_description positional_options;
        positional_options.add("type", 1);
        positional_options.add("positionsource", 1);
        positional_options.add("sink", 1);
        
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
            std::cout << "Simple-Tracker Position Filter, version 1.0\n"; //TODO: Cmake managed versioning
            std::cout << "Written by Jonathan P. Newman in the MWL@MIT.\n";
            std::cout << "Licensed under the GPL3.0.\n";
            return 0;
        }

        if (!variable_map.count("positionsource")) {
            printUsage(visible_options);
            std::cout << "Error: a position SOURCE must be specified. Exiting.\n";
            return -1;
        }

        if (!variable_map.count("sink")) {
            printUsage(visible_options);
            std::cout << "Error: a position SINK must be specified. Exiting.\n";
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
        std::cerr << "Exception of unknown type! " << std::endl;
    }

    // Make the viewer
    PositionFilter* position_filter;
    switch (type_hash[type]) {
        case 'a':
        {
            position_filter = new KalmanFilter2D(source, sink);
            break;
        }
        default:
        {
            printUsage(visible_options);
            std::cout << "Error: invalid TYPE specified. Exiting.\n";
            return -1;
        }
    }
    
    if (config_used)
        position_filter->configure(config_file, config_key);

    std::cout << "Position Filter named \"" + sink + "\" has started.\n";
    std::cout << "COMMANDS:\n";
    std::cout << "  t: Enable tuning mode.\n";
    std::cout << "  T: Disable tuning mode.\n";
    std::cout << "  x: Exit.\n";

    // Two threads - one for user interaction, the other
    // for executing the processor
    boost::thread_group thread_group;
    thread_group.create_thread(boost::bind(&run, position_filter));
    sleep(1);

    // Start the user interface
    while (!done) {

        char user_input;
        std::cin >> user_input;

        switch (user_input) {
            case 't':
            {
                position_filter->set_tune_mode(true);
                break;
            }
            case 'T':
            {
                position_filter->set_tune_mode(false);
                break;
            }
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

    // Exit gracefully and ensure all shared resources are cleaned up
    thread_group.join_all();

    // Deallocate heap
    delete position_filter;
    
    // Exit
    return 0;
}
