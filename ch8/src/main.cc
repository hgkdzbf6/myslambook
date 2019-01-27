#include <iostream> 
#include <ctime>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <list>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>

using namespace std;
using namespace cv;

#define MATRIX_SIZE 50

int main(int argc, char const *argv[])
{
    string base_path = "../data/";
    string associate_file = base_path + "associate.txt";
    ifstream fin(associate_file);
    string rgb_file, depth_file, time_rgb, time_depth;
    list<cv::Point2f> keypoints;
    cv::Mat color, depth, last_color;
    for(int index = 0;index<100;index++){
        fin>>time_rgb >> rgb_file >>time_depth>>depth_file;
        color = cv::imread(base_path + rgb_file);
        depth = cv::imread(base_path+depth_file);
        if(index == 0){
            vector<cv::KeyPoint> kps;
            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
            detector->detect(color,kps);
            for(auto kp:kps){
                keypoints.push_back(kp.pt);
            }
            last_color = color;
            continue;
        }
        if(color.data == nullptr || depth.data==nullptr)
            continue;
        vector<cv::Point2f> next_keypoints;
        vector<cv::Point2f> prev_keypoints;
        for(auto kp:keypoints)
            prev_keypoints.push_back(kp);
        vector<unsigned char> status;
        vector<float> error;
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

        cv::calcOpticalFlowPyrLK(last_color, color, prev_keypoints, next_keypoints, status, error);
        // 参数：
        // 1. 前一帧的单精度通道输入图像
        // 2. 下一帧图像的输入相同大小和相同类型的图像
        // 3. 光流计算的，和prev具有相同大小的浮点数keypoint向量。
        // 4. 下一帧图像的特征点
        // 5. 状态值，如果是0的话则说明没有找到对应的光流匹配点，是1说明找到了
        // 6. 误差
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
        cout << "LK Flow use time: " << time_used.count() << endl;
        // 把跟丢的点丢掉
        int i=0;
        for(auto iter = keypoints.begin(); iter!=keypoints.end();i++){
            if(status[i]==0){
                iter = keypoints.erase(iter);
                continue;
            }
            *iter = next_keypoints[i];
            iter++;
        }
        cout << "tracked keypoints: " << keypoints.size() << endl;
        if(keypoints.size()==0){
            cout << "all keypoints are lost." << endl;
            break;
        }
        cv::Mat img_show = color.clone();
        for(auto kp:keypoints){
            cv::circle(img_show, kp,10,cv::Scalar(0,240,0),1);
        }
        cv::imshow("corners", img_show);
        waitKey(0);
        last_color = color;
    }
    return 0;
}
