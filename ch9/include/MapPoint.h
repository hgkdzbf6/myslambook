#pragma once
#include "common.h"
#include "Frame.h"

namespace zbf{
    class MapPoint{
    public:
        typedef std::shared_ptr<MapPoint> Ptr;
        static unsigned long factory_id_;
        bool good_;

        unsigned long id_;
        Eigen::Vector3d pos_;
        Eigen::Vector3d norm_;
        cv::Mat descriptor_;

        list<Frame*> observed_frames_;

        int matched_times_;
        int visible_times_;

        MapPoint();
        MapPoint(
            unsigned long id, 
            const Eigen::Vector3d& position, 
            const Eigen::Vector3d& norm,
            Frame* frame=nullptr,
            const Mat& descriptor=Mat()
        );
        inline cv::Point3f getPositionCV() const{
            return cv::Point3f(pos_(0,0), pos_(1,0), pos_(2,0));
        }
        static MapPoint::Ptr createMapPoint();
        static MapPoint::Ptr createMapPoint(
            const Vector3d& pos_world,
            const Vector3d& norm_,
            const Mat& descriptor, 
            Frame* frame
        );
    };
}