/* 
 * File:   PositionFilter.h
 * Author: Jon Newman <jpnewman snail mit dot edu>
 *
 * Created on April 21, 2015, 9:20 PM
 */

#ifndef POSITIONFILTER_H
#define	POSITIONFILTER_H

#include "../../lib/shmem/SMServer.h"
#include "../../lib/shmem/SMClient.h"
#include "../../lib/shmem/Position.h"

class PositionFilter {
public:

    PositionFilter(std::string position_source_name, std::string position_sink_name) :
    name(position_sink_name)
    , position_source(position_source_name)
    , position_sink(position_sink_name) {

        position_source.findSharedObject();
    }

    // Position Filters must be able to grab the current position from
    // a position source (such as a Detector)
    virtual void grabPosition(void) = 0;

    // Position Filters must be able filter the position 
    virtual void filterPosition(void) = 0;

    // Position filters must be able serve the filtered position
    virtual void serveFilteredPosition(void) = 0;

    void stop(void) {
        position_source.notifySelf();
        position_sink.set_running(false);
    }

protected:

    std::string name;
    shmem::SMClient<shmem::Position> position_source;
    shmem::SMServer<shmem::Position> position_sink;
};

#endif	/* POSITIONFILTER_H */

