//******************************************************************************
//* File:   Helpers.h
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

#ifndef OAT_SHMEMDFHELPERS_H
#define	OAT_SHMEMDFHELPERS_H

#include "Source.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>

#include "../utility/in_place.h"
#include "../utility/make_unique.h"

namespace oat {

/**
 * @brief Maybe useful for tagging the T in source<T>. Not implemented.
 */
enum class TokenType : int {
    Any = -1,
    Frame,
    Pose
};

/**
 * @brief Type erased source. Not yet used.
 */
class GenericSource {

public:

    template <typename T>
    explicit GenericSource(oat::in_place<T>)
    : source(std::make_shared<SourceModel<T>>()) { }

    void touch(const std::string &addr) const { source->touch(addr); }
    void connect(void) const { source->connect(); }
    Node::State wait(void) const { return source->wait(); }
    void post(void) const { source->post(); }
    uint64_t write_number(void) const { return source->write_number(); }
    void * retrieve(void) const { return source->retrieve(); }
    double sample_period_sec(void) const { return source->sample_period_sec(); }

private:

    struct SourceConcept {
        virtual ~SourceConcept() { }
        virtual void touch(const std::string &addr) = 0;
        virtual void connect(void) = 0;
        virtual Node::State wait(void) = 0;
        virtual void post(void) = 0;
        virtual uint64_t write_number(void) const = 0;
        virtual void * retrieve(void) const = 0;
        virtual double sample_period_sec(void) const = 0;
    };

    template <typename T>
    struct SourceModel : SourceConcept {
        SourceModel() : source() { }
        void touch(const std::string &addr) override { source.touch(addr); }
        void connect(void) override { source.connect(); }
        Node::State wait(void) override { return source.wait();}
        void post(void) override { source.post(); }
        uint64_t write_number() const override { return source.write_number(); }
        void * retrieve() const override { return (void *)(source.retrieve()); }
        double sample_period_sec() const override
            { return source.retrieve()->sample().period_sec().count();}

    private:
        T source;
    };

    std::shared_ptr<SourceConcept> source;
};

/**
 * @brief Struct containing address, token type and generic source.
 * @note Not used and probably not very useful -- token type should be
 * retrievable from source without knowing anything about it except its
 * address.
 */
struct TaggedSource {

    template <typename T>
    TaggedSource(const oat::TokenType source_type,
                 const std::string &source_addr,
                 oat::in_place<T>)
    : type(source_type)
    , addr(source_addr)
    , source(oat::in_place<T>())
    {
        // Nothing
    }

    const oat::TokenType type {oat::TokenType::Any};
    const std::string addr;
    oat::GenericSource source;
};

template<typename T>
struct NamedSource {

    explicit NamedSource(const std::string &name,
                         std::unique_ptr<oat::Source<T>> &&source)
    : name(name)
    , source(std::move(source))
    {
        // Nothing
    }

    const std::string name;
    std::unique_ptr<oat::Source<T>> source;
};

template<typename T>
using NamedSourceList = std::vector<NamedSource<T>>;

/**
 * @brief Check if a set of sample periods is consistent.
 * @param periods_sec Sample periods in seconds.
 * @param min_rate The minimal sample rate in the set.
 * @param epsilon Equality tolerance.
 * @return True if maximum difference between periods is within epsilon. False
 * otherwise.
 */
inline bool checkSamplePeriods(const std::vector<double> &periods_sec,
                               double &min_rate,
                               double const epsilon = 1e-6)
{

    assert(periods_sec.size() > 0);

    double max_period = *std::max_element(std::begin(periods_sec),
                                          std::end(periods_sec));
    min_rate = 1.0 / max_period;

    if (periods_sec.size() > 1) {
        for (auto &p : periods_sec)
            if (std::fabs(p - max_period) > epsilon)
                return false;
    }

    return true;
}

inline std::string inconsistentSampleRateWarning(double min_rate)
{
    return
        "WARNING: sample rates of sources are inconsistent.\n"
        " (1) This component forces synchronization at the lowest\n"
        "     source sample rate, which is " + std::to_string(min_rate) + "\n"
        " (2) You should probably use separate components to\n"
        "     process these sources if you want independent rates.\n";
}

}       /* namespace oat */
#endif	/* OAT_SHMEMDFHELPERS_H */
