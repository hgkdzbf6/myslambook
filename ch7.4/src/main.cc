#include <iostream> 
#include <ctime>
#include <iomanip>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>


#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

using namespace std;
using namespace cv;

#define MATRIX_SIZE 50

void find_feature_matches(
    const Mat& img1, const Mat& img2,
    std::vector<KeyPoint>& k1,
    std::vector<KeyPoint>& k2,
    std::vector<DMatch>& matches
){    
    //-- 初始化
    Mat d1, d2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img1,k1 );
    detector->detect ( img2,k2 );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img1, k1, d1 );
    descriptor->compute ( img2, k2, d2 );

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( d1, d2, match );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < d1.rows; i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < d1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            matches.push_back ( match[i] );
        }
    }
}

Point2d pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2d
           (
               ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
               ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
           );
}

void bundleAdjustment(const vector<Point3f> p3d, 
const vector<Point2f> p2d, const Mat& K, Mat& R, 
Mat& t)
{
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3> > MyBlock;
    typedef g2o::LinearSolverCSparse<MyBlock::PoseMatrixType> MySolver;
    // pose 6自由度， landmark 3自由度
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg
    (g2o::make_unique<MyBlock>(g2o::make_unique<MySolver>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    //添加顶点
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    Eigen::Matrix3d R_mat;
    R_mat << 
        R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), 
        R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), 
        R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);
    pose->setId(0);
    pose->setEstimate(
        g2o::SE3Quat(
            R_mat, Eigen::Vector3d(
                t.at<double>(0,0), 
                t.at<double>(1,0), 
                t.at<double>(2,0)
            )
        )
    );
    optimizer.addVertex(pose);

    int index=1;
    for(const Point3f p: p3d){
        g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
        point->setId(index++);
        point->setEstimate(Eigen::Vector3d(p.x, p.y, p.z));
        point->setMarginalized(true);
        optimizer.addVertex(point);
    }

    // 相机内参
    g2o::CameraParameters* camera =new g2o::CameraParameters(
        K.at<double>(0,0),
        Eigen::Vector2d(K.at<double>(0,2),
        K.at<double>(1,2)),
        0
    );
    camera->setId(0);
    optimizer.addParameter(camera);

    index=1;
    for(const Point2f p:p2d){
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId(index);
        // 一头链接路标点。p3d和p2d是一一对应的
        edge->setVertex(0,dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(index)));
        // 一头连接摄像头的位置
        edge->setVertex(1,pose);
        edge->setMeasurement(Eigen::Vector2d(p.x,p.y));
        edge->setParameterId(0,0);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        index++;
    }

    // 计时
    chrono::steady_clock::time_point t1=chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>> (t2-t1);
    cout << "optimization costs time: " << time_used.count() << endl;
    cout << endl<<"after optimization: " <<endl;
    cout << "T= " << endl<<Eigen::Isometry3d(pose->estimate() ).matrix()<<endl;
}

int main(int argc, char const *argv[])
{
    Mat img_1 = imread("../pics/1.png", -1);
    Mat img_2 = imread("../pics/2.png", -1);

    Mat d1 = imread("../pics/1_depth.png",-1);
    Mat d2 = imread("../pics/2_depth.png",-1);
    vector<KeyPoint> k1,k2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, k1,k2,matches);
 
    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    vector<Point3f> p3d;
    vector<Point2f> p2d;
    for(DMatch  m:matches){
        ushort d = d1.ptr<unsigned short> 
        (int ( k1[m.queryIdx].pt.y )) 
        [ int ( k1[m.queryIdx].pt.x ) ];
        if(d==0) continue;
        float dd = d/5000.0;
        Point2d p1 = pixel2cam(k1[m.queryIdx].pt,K);
        p3d.push_back(Point3f(p1.x*dd, p1.y*dd, dd));
        p2d.push_back(k2[m.trainIdx].pt);
    }

    cout << "3d-2d pairs: " << p3d.size() << endl;

    Mat r,t;
    solvePnP(p3d,p2d,K,Mat(),r,t,false,cv::SOLVEPNP_EPNP);
    Mat R;
    Rodrigues(r,R);

    cout<<"R="<<endl<<R<<endl;
    cout<<"t="<<endl<<t<<endl;

    cout<<"calling bundle adjustment"<<endl;

    bundleAdjustment(p3d, p2d,K,R,t);
    return 0;
}
