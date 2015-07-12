//******************************************************************************
//* File:   OatTOMLSanitize.h 
//* Author: Jon Newman <jpnewman snail mit dot edu>
//*
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

#ifndef OATTOMLSANATIZE_H
#define OATTOMLSANATIZE_H

#include <string>
#include <typeinfo>
#include <boost/core/demangle.hpp>

#include "cpptoml.h"

namespace oat{
namespace config {

// Aliases for TOML table and array types
using Value = std::shared_ptr<cpptoml::base>;
using Table = std::shared_ptr<cpptoml::table>;
using Array = std::shared_ptr<cpptoml::array>;

inline void checkKeys(const std::vector<std::string>& options, const Table user_config) {

    auto it = user_config->begin();

    // Look through user-provided config options and make sure each matches an 
    // actual configuration key ID
    while (it != user_config->end()) {

        auto key = it->first;

        if (std::find(std::begin(options), std::end(options), key) == options.end()) {
            throw (std::runtime_error("Unknown configuration key '" + key + "'.\n"));
        }

        it++;
    }
}

// Nested table with unspecified number of elements
inline bool getTable(const Table table,
                const std::string& key,
                Table& nested_table) {

    // If the key is in the table,
    if (table->contains(key)) {

        // Make sure the key points to a value (and not a table, array, or table-array)
        if (table->get(key)->is_table()) {

            // Get value
            nested_table = table->get_table(key);
            return true;

        } else {
            throw (std::runtime_error("'" + key + "' must be a TOML table.\n"));
        }

    } else {
        return false;
    }
}

// Single value type sanitization
template <typename T>
bool getValue(const Table table, 
        const std::string& key, 
        T& value, bool 
        required = false) {

    // If the key is in the table,
    if (table->contains(key)) {

        // Make sure the key points to a value (and not a table, array, or table-array)
        if (table->get(key)->is_value()) {

            // Get value
            value = *table->get_as<T>(key);
            return true;

        } else {
            throw (std::runtime_error("'" + key + "' must be a TOML value of type "
                    + boost::core::demangle(typeid(T).name()) + ".\n"));
        }

    } else if (required) {
         throw (std::runtime_error("Required configuration value '" + key + "' was not specified.\n"));
    } else {
        return false;
    }
}


// Single bounded value type sanitization
template <typename T>
bool getValue(const Table table, 
        const std::string& key, 
        T& value, const T lower, 
        bool required = false) {

    // If the key is in the table,
    if (table->contains(key)) {

        // Make sure the key points to a value (and not a table, array, or table-array)
        if (table->get(key)->is_value()) {

            // Get value, check type
            auto val = *table->get_as<T>(key);
            if (val < lower) {
                 throw (std::runtime_error("Configuration key '" + key + 
                         "' specifies a value that is out of bounds.\n"));
            }

            value = val;
            return true;

        } else {
            throw (std::runtime_error("'" + key + "' must be a TOML value of type " 
                    + boost::core::demangle(typeid(T).name()) + ".\n"));
        }
    } else if (required) {
         throw (std::runtime_error("Required configuration value '" + key + "' was not specified.\n"));
    } else {
        return false;
    }
}

// Single bounded value type sanitization
template <typename T>
bool getValue(const Table table, 
        const std::string& key, 
        T& value, const T lower, 
        const T upper, 
        bool required = false) {

    // If the key is in the table,
    if (table->contains(key)) {

        // Make sure the key points to a value (and not a table, array, or table-array)
        if (table->get(key)->is_value()) {

            // Get value, check type
            auto val = *table->get_as<T>(key);
            if (val < lower || val > upper) {
                throw (std::runtime_error("Configuration key '" + key +
                        "' specifies a value that is out of bounds.\n"));
            }

            value = val;
            return true;

        } else {
            throw (std::runtime_error("'" + key + "' must be a TOML value of type "
                    + boost::core::demangle(typeid(T).name()) + ".\n"));
        }
    } else if (required) {
         throw (std::runtime_error("Required configuration value '" + key + "' was not specified.\n"));
    } else {
        return false;
    }
}

// TOML array from table, any size
inline bool getArray(const Table table, const std::string& key, Array& array_out, bool required = false) {

    // If the key is in the table,
    if (table->contains(key)) {

        // Make sure the key points to a value (and not a table, array, or table-array)
        if (table->get(key)->is_array()) {
            
            array_out = table->get_array(key);
            return true;

        } else {
            throw (std::runtime_error("'" + key + "' must be a TOML array.\n"));
        }
    } else if (required) {
         throw (std::runtime_error("Required configuration value '" + key + "' was not specified.\n"));
    } else {
        return false;
    }
}

// TOML array from table, required size
inline bool getArray(const Table table, const std::string& key, Array& array_out, int size, bool required = false) {

    // If the key is in the table,
    if (table->contains(key)) {

        // Make sure the key points to a value (and not a table, array, or table-array)
        if (table->get(key)->is_array()) {

            array_out = table->get_array(key);

            if (array_out->get().size() != size) {
                throw (std::runtime_error("'" + key + "' must be a TOML vector "
                        "containing " + std::to_string(size) + " elements.\n"));
            }

            return true;

        } else {
            throw (std::runtime_error("'" + key + "' must be a TOML vector "
                    "containing " + std::to_string(size) + " elements.\n"));
        }
    } else if (required) {
        throw (std::runtime_error("Required configuration value '" + key + "' was not specified.\n"));
    } else {
        return false;
    }
}

} // namespace config
} // namespace oat

#endif // OATTOMLSANATIZE_H
