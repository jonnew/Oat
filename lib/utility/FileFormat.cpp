//******************************************************************************
//* File:   FileFormat.cpp
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

#include <ctime>
#include <chrono>
#include <string>
#include <boost/filesystem.hpp>
#include <iostream>
#include "FileFormat.h"

namespace oat {

namespace bfs = boost::filesystem;

int createSavePath(std::string &save_path_result,
                   const std::string &save_directory,
                   const std::string &base_file_name,
                   const std::string &prepend_str,
                   const bool use_prepend_str) {

    // First check that the save_directory is valid
    bfs::path path(save_directory.c_str());
    if (!bfs::exists(path) || !bfs::is_directory(path))
        return 1;

    // Make sure there is a file_name
    if (base_file_name.empty())
        return 3;

    // Now that the directory has been confirmed, created file name
    // Create a single position file
    if (use_prepend_str)
        save_path_result = save_directory + "/" + prepend_str + base_file_name;
    else
        save_path_result = save_directory + "/" + base_file_name;

    return 0;
}

std::string createTimeStamp(bool use_msec) {

    char buffer[100];
    auto now = std::chrono::system_clock::now();
    std::time_t raw_time = std::chrono::system_clock::to_time_t(now);
    auto time_info = std::localtime(&raw_time);
    std::strftime(buffer, sizeof buffer, "%F-%H-%M-%S", time_info);

    if (use_msec) {

        char msec_buffer[5];
        auto sec =
            std::chrono::time_point_cast<std::chrono::seconds>(now);
        auto msec =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - sec);

        snprintf(msec_buffer, sizeof msec_buffer, "-%ld", msec.count());
        strcat(buffer, msec_buffer);
    }

    return std::string(buffer);
}

int ensureUniquePath(std::string& file_path) {

    int i = 0;
    std::string original_file = file_path;

    while (bfs::exists(file_path.c_str())) {

        ++i;
        bfs::path path(original_file.c_str());
        bfs::path parent_path = path.parent_path();
        bfs::path stem = path.stem();
        bfs::path extension = path.extension();

        std::string append = "_" + std::to_string(i);
        stem += append.c_str();

        // Recreate file path
        file_path = std::string(parent_path.generic_string()) + "/" +
               std::string(stem.generic_string()) +
               std::string(extension.generic_string());
    }

    return i;
}

bool checkWritePermission(const std::string &file_path) {

    FILE *fp = fopen(file_path.c_str(), "w");
    bool can_write = fp != nullptr;
    if (can_write)
        fclose(fp);

    return  can_write;
}

} /* namespace oat */
