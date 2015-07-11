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

#include "cpptoml.h"

namespace oat{
namespace config {

// Aliases for TOML table and array types
using Value = std::shared_ptr<cpptoml::base>;
using Table = std::shared_ptr<cpptoml::table>;

#ifdef __GNUG__
#include <cstdlib>
#include <memory>
#include <cxxabi.h>

/**
 * Type ID Demangler
 * @param name Mangled type identifier
 * @return Demangled type string
 */
inline std::string demangle(const char* name) {

    int status = -4; // some arbitrary value to eliminate the compiler warning

    std::unique_ptr<char, void(*)(void*)> res {
        abi::__cxa_demangle(name, NULL, NULL, &status),
        std::free
    };

    return (status==0) ? res.get() : name ;
}

#else

/**
 * Type ID Demangler. Does nothing if not g++.
 * @param name Mangled type identifier
 * @return Demangled type string
 */
inline std::string demangle(const char* name) {
    return name;
}

#endif

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
            throw (std::runtime_error(key + " must be a TOML table.\n"));
        }

    } else {
        return false;
    }
}

// Implementation declarations
std::string demangle(const char* name);
bool getTable(const Table table, const std::string& key, Table& nested_table);

/**
 * Demangle expression's type.
 */
template <class T>
std::string type(const T& t) {

    return demangle(typeid(t).name());
}

/**
 * Demangle type's type.
 */
template <class T>
std::string type() {
    return demangle(typeid(T).name());
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
            throw (std::runtime_error(key + " must be a TOML value of type "
                    + type<T>() + ".\n"));
        }

    } else if (required) {
         throw (std::runtime_error("Required key " + key + " was not specified in.\n"));
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
            throw (std::runtime_error(key + " must be a TOML value of type " 
                    + type<T>() + ".\n"));
        }
    } else if (required) {
         throw (std::runtime_error("Required key " + key + " was not specified in.\n"));
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
            throw (std::runtime_error(key + " must be a TOML value of type "
                    + type<T>() + ".\n"));
        }
    } else if (required) {
        throw (std::runtime_error("Required key '" + key + "' was not specified in.\n"));
    } else {
        return false;
    }
}

} // namespace config
} // namespace oat

#endif // OATTOMLSANATIZE_H
