//******************************************************************************
//* File:   main.cpp
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


#include <unordered_map>
#include <signal.h>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/program_options.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>

namespace po = boost::program_options;
namespace bip = boost::interprocess;

void printUsage(po::options_description options){
    std::cout << "Usage: clean [OPTIONS]\n"
              << "   or: clean NAMES\n"
              << "Remove the named shared memory segments specified by NAMES.\n\n"
              << options << "\n";
}

int main(int argc, char *argv[]) {

    std::vector<std::string> names;

    try {

        po::options_description options("META");
        options.add_options()
                ("help", "Produce help message.")
                ("version,v", "Print version information.")
                ;

        po::options_description hidden("HIDDEN OPTIONS");
        hidden.add_options()
                ("names", po::value< std::vector<std::string> >(),
                "The names of the shared memory segments to remove.")
                ;
        po::positional_options_description positional_options;
        positional_options.add("names", -1);
         
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
            std::cout << "oat clean version 1.0\n"; //TODO: Cmake managed versioning
            std::cout << "Written by Jonathan P. Newman in the MWL@MIT.\n";
            std::cout << "Licensed under the GPL3.0.\n";
            return 0;
        }

        if (!variable_map.count("names")) {
            printUsage(all_options);
            std::cout << "Error: at least a single NAME must be specified. Exiting.\n";
            return -1;
        }
        
        names = variable_map["names"].as< std::vector<std::string> >();


    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Exception of unknown type! " << std::endl;
    }

    for (auto &name : names) {
        
        // All servers (MatServer and SMServer) append "_sh_mem" to user-provided
        // stream names when created a named shmem block
        name = name + "_sh_mem";
        
        std::cout << "Removing " << name << " from shared memory...";
        if (bip::shared_memory_object::remove(name.c_str()) ) {
            
            std::cout << "success.\n";
        } else {
            std::cout << "failure.\n";
            std::cout << "Are you sure this block exists?\n";
        }
    }

    // Exit
    return 0;
}

