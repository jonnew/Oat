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

#include <csignal>
#include <unordered_map>
#include <boost/program_options.hpp>

#include "../../lib/utility/IOFormat.h"
#include "MeanPosition2D.h"

namespace po = boost::program_options;

volatile sig_atomic_t quit = 0;
volatile sig_atomic_t source_eof = 0;

void printUsage(po::options_description options) {
    std::cout << "Usage: combiner [INFO]\n"
              << "   or: combiner TYPE SOURCES SINK\n"
              << "Combine positional information from two or more SOURCES.\n"
              << "Publish combined position to SINK.\n\n"
              << "TYPE\n"
              << "  mean: Geometric mean of SOURCE positions\n\n"
              << "SOURCES:\n"
              << "  User supplied position source names (e.g. pos1 pos2).\n\n"
              << "SINK:\n"
              << "  User supplied position sink name (e.g. pos).\n\n"
              << options << "\n";
}

// Signal handler to ensure shared resources are cleaned on exit due to ctrl-c
void sigHandler(int s) {
    quit = 1;
}

void run(PositionCombiner* combiner) {
    
    while (!quit && !source_eof) { 
        
        source_eof = combiner->process(); 
    }
}

int main(int argc, char *argv[]) {
    
    std::signal(SIGINT, sigHandler);

    std::vector<std::string> sources;
    std::string sink;
    std::string type;
    po::options_description options("INFO");
    
    std::unordered_map<std::string, char> type_hash;
    type_hash["mean"] = 'a';

    try {

        options.add_options()
                ("help", "Produce help message.")
                ("version,v", "Print version information.")
                ;
        
        po::options_description hidden("HIDDEN OPTIONS");
        hidden.add_options()
                ("type", po::value<std::string>(&type), 
                "Type of test position combiner to use.")
                ("sources", po::value< std::vector<std::string> >(),
                "The names the SOURCES supplying the Position2D objects to be combined.")
                ("sink", po::value<std::string>(&sink),
                "The name of the SINK to which combined position Position2D objects will be published.")
                ;
        

        po::positional_options_description positional_options;
        positional_options.add("type", 1);
        positional_options.add("sources", -1); // If not overridend by explicit --sink, last positional argument is sink.

        po::options_description all_options("OPTIONS");
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
        
        if (!variable_map.count("type")) {
            printUsage(options);
            std::cout << "Error: a TYPE must be specified. Exiting.\n";
            return -1;
        }
        
        if (!variable_map.count("sources")) {
            printUsage(options);
            std::cout << "Error: at least two SOURCES and a SINK must be specified. Exiting.\n";
            return -1;
        }
        sources = variable_map["sources"].as< std::vector<std::string> >();
        
        if (!variable_map.count("sink")) {
            
            // If not overridden by explicit --sink, last positional argument is the sink.
            sink = sources.back();
            sources.pop_back(); 
        }

    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Exception of unknown type!" << std::endl;
        return 1;
    }

    PositionCombiner* combiner; //(ant_source, pos_source, sink);
    switch (type_hash[type]) {
        case 'a':
        {
            combiner = new MeanPosition2D(sources, sink);
            break;
        }
        default:
        {
            printUsage(options);
            std::cout << "Error: invalid TYPE specified. Exiting.\n";
            return -1;
        }
    }
    
    // Tell user
    std::cout << oat::whoMessage(combiner->get_name(), "Listening to sources ");
    for (auto s : sources)
        std::cout << oat::sourceText(s) << " ";          
    std::cout << ".\n"
              << oat::whoMessage(combiner->get_name(),
                 "Steaming to sink " + oat::sinkText(sink) + ".\n")
              << oat::whoMessage(combiner->get_name(), 
                 "Press CTRL+C to exit.\n");
    
    // Infinite loop until ctrl-c or end of stream signal
    run(combiner);
    
    // Tell user
    std::cout << oat::whoMessage(combiner->get_name(), "Exiting.\n");

    // Free heap memory allocated to combiner
    delete combiner;

    // Exit
    return 0;
}


