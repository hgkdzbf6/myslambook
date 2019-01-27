#include "MapPoint.h"

namespace zbf{

    MapPoint::MapPoint()
    : id_(-1), pos_(Eigen::Vector3d(0,0,0)), norm_(Eigen::Vector3d(0,0,0)), matched_times_(0), visible_times_(0), descriptor_(Mat()),good_(true)
    {

    }

    MapPoint::MapPoint ( unsigned long id, const Eigen::Vector3d& position, const Eigen::Vector3d& norm, Frame* frame, const Mat& descriptor )
    : id_(id), pos_(position), norm_(norm),good_(true), matched_times_(1), visible_times_(1), descriptor_(descriptor)
    {
        observed_frames_.push_back(frame);
    }

    MapPoint::Ptr MapPoint::createMapPoint()
    {
        return MapPoint::Ptr( 
            new MapPoint( 
                factory_id_++, Eigen::Vector3d(0,0,0), Eigen::Vector3d(0,0,0)
            )
        );
    }

    MapPoint::Ptr MapPoint::createMapPoint(            
        const Vector3d& pos_world,
        const Vector3d& norm_,
        const Mat& descriptor, 
        Frame* frame=nullptr)
    {
        return MapPoint::Ptr( 
            new MapPoint( factory_id_++, pos_world, norm_ ,frame, descriptor )
        );
    }

    unsigned long MapPoint::factory_id_ = 0;

}