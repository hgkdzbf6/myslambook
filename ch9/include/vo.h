#pragma once

#include <common.h>
#include <Frame.h>
#include <Map.h>

namespace zbf{
    enum VOState{
        INITIALIZING = -1,
        OK=0, 
        LOST=1,
    };

    class VO{
    public:
        typedef std::shared_ptr<VO> Ptr;
        VOState state_;
        Map::Ptr map_;
        Frame::Ptr ref_;
        Frame::Ptr curr_;

        cv::Ptr<cv::ORB> orb_;
        std::vector<cv::Point3f> pts_3d_ref_;
        std::vector<cv::KeyPoint> keypoints_curr_;
        cv::Mat descriptors_curr_;
        cv::Mat descriptors_ref_;
        std::vector<cv::DMatch> feature_matches_;

    
        cv::FlannBasedMatcher   matcher_flann_;     // flann matcher
        vector<MapPoint::Ptr>   match_3dpts_;       // matched 3d points 
        vector<int>             match_2dkp_index_;  // matched 2d pixels (index of kp_curr)
   

        Sophus::SE3d T_c_r_estimated_;
        int num_inliers_;
        int num_lost_;

        int num_of_features_;
        double scale_factor_;
        int level_pyramid_;
        float match_ratio_;
        int max_num_lost_;
        int min_inliers_;

        double key_frame_min_rot;
        double key_frame_min_trans;
        double map_point_erase_ratio_; // remove map point ratio
   
    public:
        VO();
        ~VO();
        bool addFrame(Frame::Ptr frame);
    protected:
        void extractKeyPoints();
        void computeDescriptors();
        void featureMatching();
        void poseEstimationPnP();
        void setRef3dPoints();

        void addKeyFrame();
        bool checkEstimatedPose();
        bool checkKeyFrame();

        void optimizeMap();
        void addMapPoints();
        void squeezeMapPoints();
        double getViewAngle(Frame::Ptr frame, MapPoint::Ptr point);
    };
}