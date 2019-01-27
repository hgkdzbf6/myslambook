#pragma once
#include <common.h>
#include "Camera.h"

namespace zbf{
    class Frame{
    public:
        typedef std::shared_ptr<Frame> Ptr;
        unsigned long id_;
        double time_stamp_;
        Sophus::SE3d T_c_w_;
        Camera::Ptr camera_;
        cv::Mat color_,depth_;
    
    public:
        Frame();
        Frame(long id, double time_stamp=0, Sophus::SE3d T_c_w = Sophus::SE3d(),Camera::Ptr camera = nullptr, cv::Mat color = cv::Mat(), cv::Mat depth = cv::Mat());
        ~Frame();

        static Frame::Ptr createFrame();
        double findDepth(const cv::KeyPoint& kp);

        Eigen::Vector3d getCameraCenter()const;

        bool isInFrame( const Vector3d& pt_world);
    };
}