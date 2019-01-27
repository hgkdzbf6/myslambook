#include <vo.h>
#include <Config.h>
#include <g2o_type.h>

namespace zbf{
    VO::VO():    state_ ( INITIALIZING ), ref_ ( nullptr ), curr_ ( nullptr ), map_ ( new Map ), num_lost_ ( 0 ), num_inliers_ ( 0 ) , matcher_flann_ ( cv::makePtr<cv::flann::LshIndexParams>( 5,10,2 ))  {
        num_of_features_ = Config::get<int>("number_of_features");    
        scale_factor_       = Config::get<double> ( "scale_factor" );
        level_pyramid_      = Config::get<int> ( "level_pyramid" );
        match_ratio_        = Config::get<float> ( "match_ratio" );
        max_num_lost_       = Config::get<float> ( "max_num_lost" );
        min_inliers_        = Config::get<int> ( "min_inliers" );
        key_frame_min_rot   = Config::get<double> ( "keyframe_rotation" );
        key_frame_min_trans = Config::get<double> ( "keyframe_translation" );
        map_point_erase_ratio_ = Config::get<double> ( "map_point_erase_ratio" );
        orb_ = cv::ORB::create ( num_of_features_, scale_factor_, level_pyramid_ );

    }

    VO::~VO(){}

    bool VO::addFrame(Frame::Ptr frame){
        switch(state_){
            case INITIALIZING:
            {
                state_ = OK;
                curr_ =ref_= frame;
                // ref_ = frame;
                // map_->insertKeyFrame ( frame );        
                extractKeyPoints();
                computeDescriptors();
                // setRef3dPoints();
                addKeyFrame();
                break;
            }
            case OK:
            {
                curr_ = frame;
                curr_->T_c_w_ = ref_->T_c_w_;
                extractKeyPoints();
                computeDescriptors();
                featureMatching();
                poseEstimationPnP();
                if(checkEstimatedPose() == true){
                    curr_->T_c_w_ = T_c_r_estimated_;
                    optimizeMap();
                    num_lost_ = 0;
                    if(checkKeyFrame() == true){
                        addKeyFrame();
                    }
                }else{
                    num_lost_++;
                    if(num_lost_>max_num_lost_){
                        state_ = LOST;
                    }
                    return false;
                }
                break;
            }
            case LOST:
            {
                std::cout<<"vo has lost."<<std::endl;
                break;

            }
        }
    }

    void VO::extractKeyPoints(){
        orb_->detect(curr_->color_, keypoints_curr_);
        cout << "keypoints size: " << keypoints_curr_.size() << endl;

    }

    void VO::computeDescriptors(){
        orb_->compute(curr_->color_, keypoints_curr_, descriptors_curr_);
        cout << "descriptors size: "<< descriptors_curr_.size() << endl;
    }

    void VO::featureMatching(){
        boost::timer timer;
        vector<cv::DMatch> matches;
        Mat desp_map;
        vector<MapPoint::Ptr> candidate;
        for(auto& all_points: map_->map_points_){
            MapPoint::Ptr& p = all_points.second;
            if(curr_->isInFrame(p->pos_)){
                p->visible_times_++;
                candidate.push_back(p);
                desp_map.push_back(p->descriptor_);
            }
        }
        matcher_flann_.match(desp_map,descriptors_curr_, matches);
        float min_dist = std::min_element(
            matches.begin(),matches.end(),[](const cv::DMatch& m1, const cv::DMatch& m2){
                return m1.distance < m2.distance;
            }
        )->distance;

        match_3dpts_.clear();
        match_2dkp_index_.clear();
        for(cv::DMatch& m:matches){
            if(m.distance< max<float>(min_dist*match_ratio_,30)){
                match_3dpts_.push_back(candidate[m.queryIdx]);
                match_2dkp_index_.push_back(m.trainIdx);
            }
        }
        cout<<"good matches: "<<match_3dpts_.size() <<endl;
        cout<<"match cost time: "<<timer.elapsed() <<endl;

    
    }
    void VO::setRef3dPoints(){
        pts_3d_ref_.clear();
        descriptors_ref_ = cv::Mat();
        int amount = 0;
        for(size_t i=0;i<keypoints_curr_.size();i++){
            double d = ref_->findDepth(keypoints_curr_[i]);

            if(d>0){
                // std::cout << "d: " << d <<  std::endl;
                Eigen::Vector3d p_cam = ref_->camera_->pixel2camera(Eigen::Vector2d(keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y), d);
                pts_3d_ref_.push_back(cv::Point3f(p_cam(0, 0), p_cam(1,0), p_cam(2,0)));
                descriptors_ref_.push_back(descriptors_curr_.row(i));
                amount ++;
            }
        }
        std::cout << "d: " << amount << std::endl;
    }

    void VO::poseEstimationPnP(){

        // construct the 3d 2d observations
        std::vector<cv::Point3f> pts3d;
        std::vector<cv::Point2f> pts2d;

        for(int index:match_2dkp_index_){
            pts2d.push_back(keypoints_curr_[index].pt);
        }
        for(MapPoint::Ptr pt:match_3dpts_){
            pts3d.push_back(pt->getPositionCV());
        }
        cv::Mat K = ( cv::Mat_<double>(3,3)<<
            ref_->camera_->fx_, 0, ref_->camera_->cx_,
            0, ref_->camera_->fy_, ref_->camera_->cy_,
            0,0,1
        );

        cv::Mat rvec, tvec, inliers;
        cv::solvePnPRansac( pts3d, pts2d, K, cv::Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers );
        // std::cout << "pnp ransac: " <<  inliers.rows << "," << inliers.cols <<  std::endl;
        num_inliers_ = inliers.rows;
        // std::cout << "PnP inliers: " << num_inliers_ << std::endl;
        T_c_r_estimated_ = SE3d(
            Sophus::SO3d::exp (Sophus::Vector3d(rvec.at<double> ( 0,0 ), rvec.at<double> ( 1,0 ), rvec.at<double> ( 2,0 ) ) ),
            Vector3d ( tvec.at<double> ( 0,0 ), tvec.at<double> ( 1,0 ), tvec.at<double> ( 2,0 ) )
        );
        typedef BlockSolver<BlockSolverTraits<6,3>> MyBlock;
        typedef LinearSolverDense<MyBlock::PoseMatrixType> MySolver;

        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<MyBlock>(g2o::make_unique<MySolver>()));
        
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
        pose->setId( 0 );
        pose->setEstimate(g2o::SE3Quat(
            T_c_r_estimated_.rotationMatrix(), T_c_r_estimated_.translation()
        ));
        optimizer.addVertex(pose);
        for(int i=0;i<inliers.rows;i++){
            int index = inliers.at<int>(i,0);
            EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly();
            edge->setId(i);
            edge->setVertex(0,pose);
            edge->camera_ = curr_->camera_.get();
            edge->point_ = Vector3d(pts3d[index].x, pts3d[index].y, pts3d[index].z);
            edge->setMeasurement(Vector2d(pts2d[index].x, pts2d[index].y));
            edge->setInformation(Eigen::Matrix2d::Identity());
            optimizer.addEdge(edge);
        }
        optimizer.initializeOptimization();
        optimizer.optimize(10);

        T_c_r_estimated_ = Sophus::SE3d(
            pose->estimate().rotation(),
            pose->estimate().translation()
        );
    }
    bool VO::checkEstimatedPose(){
        if(num_inliers_ < min_inliers_){
            std::cout<<"reject because inlier is too small: "<<num_inliers_<<std::endl;
            return false;
        }
        Sophus::Vector6d d = T_c_r_estimated_.log();
        if(d.norm()>5.0){
            std::cout<<"reject because motion is too large: "<<d.norm()<<std::endl;
            return false;
        }
        return true;
    }

    bool VO::checkKeyFrame()
    {
        Sophus::Vector6d d = T_c_r_estimated_.log();
        Vector3d trans = d.head<3>();
        Vector3d rot = d.tail<3>();
        if ( rot.norm() >key_frame_min_rot || trans.norm() >key_frame_min_trans )
            return true;
        return false;
    }


    void VO::addKeyFrame()
    {
        if(map_->keyframes_.empty()){
            for(size_t i=0;i<keypoints_curr_.size();i++){
                double d = curr_->findDepth(keypoints_curr_[i]);
                if(d<0)continue;
                Vector3d p_world = ref_->camera_->pixel2world(
                    Vector2d(keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y), curr_->T_c_w_,d
                );
                Vector3d n = p_world - ref_->getCameraCenter();
                n.normalize();
                MapPoint::Ptr map_point = MapPoint::createMapPoint(
                    p_world, n, descriptors_curr_.row(i).clone(), curr_.get()
                );
                map_->insertMapPoint(map_point);
            }
        }
        map_->insertKeyFrame(curr_);
        ref_ = curr_;
    }

    void VO::optimizeMap(){
        int not_in_frame = 0;
        int ratio_erase = 0;
        int angle_num = 0;
        for(auto it = map_->map_points_.begin(); it!=map_->map_points_.end();){
            if(!curr_->isInFrame(it->second->pos_)){
                not_in_frame++;
                // assert(not_in_frame<2000);
                it = map_->map_points_.erase(it);
                continue;
            }
            float match_ratio = (float)(it->second->matched_times_)/(it->second->visible_times_);
            if(match_ratio_<map_point_erase_ratio_){
                ratio_erase++;
                it = map_->map_points_.erase(it);
                continue;
            }
            double angle = getViewAngle(curr_, it->second);
            if(angle>M_PI/6.0){
                angle_num++;
                it=map_->map_points_.erase(it);
                continue;
            }
            if(it->second->good_==false){
                // 三角优化
            }
            it++;
        }
        if( match_2dkp_index_.size()<100){
            addMapPoints();
        }
        if( map_->map_points_.size()>1000){
            // TODO 地图太大了，要移除一点
            // map_point_erase_ratio_=0.05;
        }else{
            // map_point_erase_ratio_ = 0.1;
        }
        if(map_->map_points_.size()!=0){
            cout << "map points: " << map_->map_points_.size() << endl;
        }else{
            cout << "not in frame: " << not_in_frame
            << " ratio erase: " << ratio_erase
            << " angle num: " << angle_num
            << endl;
        }
    }

    double VO::getViewAngle(Frame::Ptr frame, MapPoint::Ptr point){
        Vector3d n = point->pos_-frame->getCameraCenter();
        n.normalize();
        return acos(n.transpose()*point->norm_);
    }

    void VO::addMapPoints(){
        vector<bool> matched(keypoints_curr_.size(), false);
        for( int index:match_2dkp_index_ ){
            matched[index] = true;
        }
        for(int i=0;i<keypoints_curr_.size();i++){
            if(matched[i] == true){
                continue;
            }
            double d = ref_->findDepth( keypoints_curr_[i]);
            if(d<0)continue;
            Vector3d p_world = ref_->camera_->pixel2world(
                Vector2d(keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y),
                curr_->T_c_w_, d
            );
            Vector3d n = p_world - ref_->getCameraCenter();
            n.normalize();
            MapPoint::Ptr map_point = MapPoint::createMapPoint(p_world, n,descriptors_curr_.row(i).clone(), curr_.get());
            map_->insertMapPoint(map_point);
        }
    }

    void VO::squeezeMapPoints(){
        int erase_num = map_point_erase_ratio_* map_->map_points_.size();
        for(auto it = map_->map_points_.begin(); it!=map_->map_points_.end();){
            it->second->id_;
            it++;
        }
    }
}