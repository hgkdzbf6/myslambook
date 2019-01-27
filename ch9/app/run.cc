#include <fstream>
#include <common.h>
#include <Config.h>
#include <vo.h>
#include <boost/timer.hpp>

int main(int argc, char const *argv[])
{
    /* code */
    zbf::Config::setParameterFile("../config/config.yaml");
    zbf::VO::Ptr vo(new zbf::VO);

    std::string dataset_dir = zbf::Config::get<std::string>("dataset_dir");
        
    std::cout<<"dataset: "<<dataset_dir<<std::endl;

    std::ifstream fin(dataset_dir+"/associate.txt");
    if(!fin){
        std::cout << "please generate the associate file called associate.txt" << std::endl;
        return 1;
    }
    std::vector<string> rgb_files, depth_files;
    vector<double> rgb_times, depth_times;
    while(!fin.eof()){
        string rgb_time, rgb_file, depth_time, depth_file;
        fin>> rgb_time>> rgb_file>>depth_time>>depth_file;
        rgb_times.push_back(atof(rgb_time.c_str()));
        depth_times.push_back(atof(depth_time.c_str()));
        rgb_files.push_back ( dataset_dir+"/"+rgb_file );
        depth_files.push_back ( dataset_dir+"/"+depth_file );

        if ( fin.good() == false )
            break;
    }
    zbf::Camera ::Ptr camera( new zbf::Camera(
        zbf::Config::get<float>("camera.fx"),
        zbf::Config::get<float>("camera.fy"),
        zbf::Config::get<float>("camera.cx"),
        zbf::Config::get<float>("camera.cy"),
        zbf::Config::get<float>("camera.depth_scale")
    ));
    
    cv::viz::Viz3d vis("Visual Odometry");
    cv::viz::WCoordinateSystem world_coor(1.0), camera_coor(0.5);
    cv::Point3d cam_pos(0,-1.0,-1.0), cam_focal_point(0,0,0), cam_y_dir(0,1,0);
    cv::Affine3d cam_pose = cv::viz::makeCameraPose( cam_pos, cam_focal_point, cam_y_dir);
    vis.setViewerPose(cam_pose);

    world_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
    camera_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 1.0);
    vis.showWidget("World", world_coor);
    vis.showWidget("Camera", camera_coor);
    cout << "read total " << rgb_files.size() << " entries" << endl;
    for( int i=0;i<rgb_files.size();i++){
        cout << "**************entry  " <<i << " **************" << endl;
        Mat color = cv::imread(rgb_files[i]);
        Mat depth = cv::imread(depth_files[i], -1);
        if(color.data == nullptr|| depth.data == nullptr){
            break; 
        }
        zbf::Frame::Ptr pFrame = zbf::Frame::createFrame();
        pFrame->camera_ = camera;
        pFrame->color_ = color;
        pFrame->depth_ = depth;
        pFrame->time_stamp_ = rgb_times[i];

        boost::timer timer;
        vo->addFrame(pFrame);
        cout << "VO costs time: " << timer.elapsed() << endl;

        if(vo->state_ == zbf::VOState::LOST){
            cout << "LOST, exit" << endl;
            break;
        }
        SE3d Tcw= pFrame->T_c_w_.inverse();

        cv::Affine3d M(
            cv::Affine3d::Mat3(
                Tcw.rotationMatrix()(0,0), Tcw.rotationMatrix()(0,1), Tcw.rotationMatrix()(0,2),
                Tcw.rotationMatrix()(1,0), Tcw.rotationMatrix()(1,1), Tcw.rotationMatrix()(1,2),
                Tcw.rotationMatrix()(2,0), Tcw.rotationMatrix()(2,1), Tcw.rotationMatrix()(1,2)
            ),
            cv::Affine3d::Vec3(
                Tcw.translation()(0,0), 
                Tcw.translation()(1,0), 
                Tcw.translation()(2,0)
            )
        );
        Mat img_show = color.clone();
        for(auto& pt:vo->map_->map_points_){
            zbf::MapPoint::Ptr p = pt.second;
            Vector2d pixel = pFrame->camera_->world2pixel(p->pos_, pFrame->T_c_w_);
            cv::circle(img_show, cv::Point2f(pixel(0,0), pixel(1,0)), 5, cv::Scalar(0,255,0),2);
        }
        cv::imshow("image", img_show);
        cv::waitKey(1);
        vis.setWidgetPose("Camera", M);
        vis.spinOnce(1,false);
    }
    return 0;
}
