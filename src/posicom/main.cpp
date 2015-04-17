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

#include "PositionCombiner.h"

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

void run(PositionCombiner* combiner) {
    
     std::cout << "Position combiner \"" + combiner->get_name() + "\" has started.\n";

    while (!done) { // !done
        combiner->calculateGeometricMean();
        combiner->serveCombinedPosition();
    }
}

void printUsage(po::options_description options) {
    std::cout << "Usage: combiner [OPTIONS]\n";
    std::cout << "   or: combiner ANTERIOR_SOURCE POSTERIOR_SOURCE SINK\n"; //TODO: N sources
    std::cout << "Combine positional information from two SMServer<Position2D> SOURCEs.\n";
    std::cout << "Publish processed object positions to a SMServer<Position2D> SINK.\n";
    std::cout << options << "\n";
}

int main(int argc, char *argv[]) {

    signal(SIGINT, term);
    
    std::string ant_source, pos_source;
    std::string sink;

    try {

        po::options_description options("OPTIONS");
        options.add_options()
                ("help", "Produce help message.")
                ("version,v", "Print version information.")
                ;
        
        po::options_description hidden("HIDDEN OPTIONS");
        hidden.add_options()
                ("anterior", po::value<std::string>(&ant_source),
                "The name of the ANTERIOR_SOURCE that supplies Position2D objects defining the anterior position.")
                ("posterior", po::value<std::string>(&pos_source),
                "The name of the POSTERIOR_SOURCE that supplies Position2D objects defining the posterior position.")
                ("sink", po::value<std::string>(&sink),
                "The name of the SINK to which combined position Position2D objects will be published.")
                ;

        po::positional_options_description positional_options;
        positional_options.add("anterior", 1);
        positional_options.add("posterior", 1);
        positional_options.add("sink", 1);

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
            std::cout << "Simple-Tracker Object Position Combiner version 1.0\n"; //TODO: Cmake managed versioning
            std::cout << "Written by Jonathan P. Newman in the MWL@MIT.\n";
            std::cout << "Licensed under the GPL3.0.\n";
            return 0;
        }

        if (!variable_map.count("anterior")) {
            printUsage(options);
            std::cout << "Error: an ANTERIOR_SOURCE must be specified. Exiting.\n";
            return -1;
        }
        
        if (!variable_map.count("posterior")) {
            printUsage(options);
            std::cout << "Error: a POSTERIOR_SOURCE must be specified. Exiting.\n";
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
        std::cerr << "Exception of unknown type!" << std::endl;
        return 1;
    }

    PositionCombiner combiner(ant_source, pos_source, sink);

    // Two threads - one for user interaction, the other
    // for executing the processor
    boost::thread_group thread_group;
    thread_group.create_thread(boost::bind(&run, &combiner));
    sleep(1);

    while (!done) {
        int user_input;
        std::cout << "Select an action:\n";
        std::cout << " [2]: Exit\n";
        std::cout << ">> ";

        std::cin >> user_input;

        switch (user_input) {
            case 2:
            {
                done = true;
                combiner.stop();
                break;
            }
            default:
                std::cout << "Invalid selection. Try again.\n";
                break;
        }
    }
    
    // TODO: Exit gracefully and ensure all shared resources are cleaned up!
    thread_group.join_all();

    // Exit
    return 0;
}


