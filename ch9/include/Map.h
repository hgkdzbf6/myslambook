#pragma once

#include "common.h"
#include "MapPoint.h"
#include "Frame.h"

namespace zbf{
    class Map{
    public:
        typedef std::shared_ptr<Map> Ptr;
        std::unordered_map<unsigned long, MapPoint::Ptr> map_points_;
        std::unordered_map<unsigned long, Frame::Ptr> keyframes_;

        Map(){}
        void insertKeyFrame(Frame::Ptr frame);
        void insertMapPoint(MapPoint::Ptr map_points);
        
    };
}