#include <iostream> 
#include <ctime>
#include <iomanip>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
using namespace std;
using namespace cv;

#define MATRIX_SIZE 50

void pose_estimation_2d2d(std::vector<KeyPoint> k1,std::vector<KeyPoint> k2,
    std::vector<DMatch> matches, Mat& R,Mat& t,Mat& E){
    Mat K = (Mat_<double>(3,3) << 356.2841797552195 ,0,333.09659026948043,
     0, 353.1672954414121, 195.25650805378962,
     0, 0, 1);
    vector<Point2f> points1;
    vector<Point2f> points2;

    for(int i=0;i<( int ) matches.size();i++){
        points1.push_back(k1[matches[i].queryIdx].pt);
        points2.push_back(k2[matches[i].trainIdx].pt);
    }

    // 计算基础矩阵
    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat( points1, points2,FM_RANSAC);
    cout << "fundamental matrix is: " <<endl<< fundamental_matrix << endl;
    // 计算本质矩阵
    Point2d principal_point (311.181370, 185.775805);
    int focal_length = 347.924711;
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point, RANSAC);
    cout << "essential matrix is: " <<endl << essential_matrix << endl;

    Mat homography_matrix;
    homography_matrix = findHomography( points1, points2, RANSAC, 3, noArray());
    cout << "homography matrix is: " <<endl << homography_matrix << endl;

    recoverPose(essential_matrix, points1, points2, R,t, focal_length,principal_point);
    E = essential_matrix.clone();

}

// 像素的2维坐标变成相机坐标的二维坐标
Point2d pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2d
           ( //(px-cx)/fx
               ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
               ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
           );
}

void triangulation(const vector<KeyPoint>& k1,
    const vector<KeyPoint>& k2, const vector<DMatch>& matches,
    const Mat& R, const Mat& t, vector<Point3d>& points){

    Mat T1 = (Mat_<double>(3,4) << 1,0,0,0,
    0,1,0,0,
    0,0,1,0);
    Mat T2 = (Mat_<double>(3,4) << 
    R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0,0), 
    R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1,0), 
    R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2,0));
    Mat K = (Mat_<double>(3,3) << 356.2841797552195 ,0,333.09659026948043,
     0, 353.1672954414121, 195.25650805378962,
     0, 0, 1);
    vector<Point2d> pt1, pt2;
    for(DMatch m:matches){
        pt1.push_back(pixel2cam(k1[m.queryIdx].pt,K));
        pt2.push_back(pixel2cam(k2[m.trainIdx].pt,K));
    }
    Mat pt4d;
    triangulatePoints(T1,T2,pt1,pt2,pt4d);

    for(int i=0;i<pt4d.cols;i++){
        Mat x = pt4d.col(i);
        x/=x.at<float>(3,0);
        Point3d p(x.at<float>(0,0),x.at<float>(1,0),x.at<float>(2,0));
        points.push_back(p);
    }
}

int main(int argc, char const *argv[])
{
    Mat img_1 = imread("../pics/left0.jpg", 0);
    Mat img_2 = imread("../pics/right0.jpg", 0);
    std::vector<KeyPoint> keypoints_1,keypoints_2;
    Mat descriptors_1, descriptors_2;
    
    Ptr<ORB> orb =  ORB::create();
    orb->detect(img_1,keypoints_1);
    orb->detect(img_2,keypoints_2);

    orb->compute(img_1, keypoints_1,descriptors_1);
    orb->compute(img_2, keypoints_2,descriptors_2);

    Mat outimg1;
    drawKeypoints(img_1,keypoints_1,outimg1,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
    imshow("orb features.", outimg1);

    vector<DMatch> matches;
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(descriptors_1, descriptors_2, matches);
    
    double min_dist=10000, max_dist = 0;
    for(int i=0;i<descriptors_1.rows;i++){
        double dist = matches[i].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }

    std::vector<DMatch> good_matches;
    for(int i=0;i<descriptors_1.rows;i++){
        if(matches[i].distance <=max(2*min_dist, 30.0)){
            good_matches.push_back(matches[i]);
        }
    }

    Mat img_match;
    Mat img_goodmatch;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2,matches, img_match);
    drawMatches(img_1, keypoints_1, img_2, keypoints_2,good_matches,img_goodmatch);
    imshow("all", img_match);
    imshow("good",img_goodmatch);
    waitKey();

    Mat R,t,E;
    pose_estimation_2d2d(keypoints_1, keypoints_2,good_matches, R,t,E);    
    cout << "R is: " <<endl << R<< endl<< "t is: " <<endl <<t << endl;

    Mat t_x = (Mat_<double>(3,3) << 
        0,          -t.at<double>(2,0), t.at<double>(1,0),
        t.at<double>(2,0),          0, -t.at<double>(0,0),
        -t.at<double>(1,0), t.at<double>(0,0),0);
    
    cout << "t^R: "  <<endl<< t_x*R <<endl;
    // 验证対极约束
    Mat K = (Mat_<double>(3,3) << 356.2841797552195 ,0,333.09659026948043,
     0, 353.1672954414121, 195.25650805378962,
     0, 0, 1);
    int i=0;
    for( auto m :good_matches) {
        Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        Mat y1 = (Mat_<double>(3,1) << pt1.x, pt1.y, 1);
        Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        Mat y2 = (Mat_<double>(3,1) << pt2.x, pt2.y, 1);
        Mat d1 = y2.t()*t_x*R*y1;
        Mat d2 = y2.t() *E* y1;
        cout << "epipolar constrain "<< i<<": " << d1 << endl;
        cout << "epipolar constrain "<< i<<": " << d2 << endl;
        i++;
    }

    vector<Point3d> points;
    triangulation(keypoints_1, keypoints_2, good_matches, R,t,points);

    for(int i=0;i<good_matches.size();i++){
        Point2d pt1 = pixel2cam(keypoints_1[good_matches[i].queryIdx].pt,K);
        Point2d pt1_3d(points[i].x / points[i].z, points[i].y / points[i].z);
        // Point2d pt2 = pixel2cam(keypoints_2[good_matches[i].trainIdx].pt,K);
        cout << "point in the first camera frame: " << pt1 << endl;
        cout << "point projected from 3D: " << pt1_3d << ", d=" << points[i].z << endl;
        Point2d pt2 = pixel2cam(keypoints_2[good_matches[i].trainIdx].pt,K);
        Mat pt2_trans = R*( Mat_<double>(3,1) << points[i].x, points[i].y, points[i].z) + t;
        pt2_trans /=pt2_trans.at<double>(2,0);
        cout << "point in the second camera frame: " << pt2 <<endl;
        cout << "point reprojected from second frame: " <<pt2_trans.t()<<endl;
        cout<<endl;
    }


    return 0;
}
