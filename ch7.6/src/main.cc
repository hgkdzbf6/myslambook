#include <iostream> 
#include <ctime>
#include <iomanip>
#include <chrono>

#include <Eigen/SVD>

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

class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3,Eigen::Vector3d, g2o::VertexSE3Expmap> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeProjectXYZRGBDPoseOnly(const Eigen::Vector3d& point):_point(point){}
    virtual void computeError(){
        const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        _error = _measurement - pose->estimate().map(_point);
    }
    virtual void linearizeOplus(){
        g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
        g2o::SE3Quat T(pose->estimate());
        Eigen::Vector3d xyz_trans = T.map(_point);
        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];

        _jacobianOplusXi(0,0) = 0;
        _jacobianOplusXi(0,1) = -z;
        _jacobianOplusXi(0,2) = y;
        _jacobianOplusXi(0,3) = -1;
        _jacobianOplusXi(0,4) = 0;
        _jacobianOplusXi(0,5) = 0;

        _jacobianOplusXi(1,0) = z;
        _jacobianOplusXi(1,1) = 0;
        _jacobianOplusXi(1,2) = -x;
        _jacobianOplusXi(1,3) = 0;
        _jacobianOplusXi(1,4) = -1;
        _jacobianOplusXi(1,5) = 0;

        _jacobianOplusXi(2,0) = -y;
        _jacobianOplusXi(2,1) = x;
        _jacobianOplusXi(2,2) = 0;
        _jacobianOplusXi(2,3) = 0;
        _jacobianOplusXi(2,4) = 0;
        _jacobianOplusXi(2,5) = -1;
    }
    virtual bool read(istream& in){}
    virtual bool write(ostream& out)const{}
protected:
    Eigen::Vector3d _point;
};

Point2d pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2d
           (
               ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
               ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
           );
}

void g2o_icp(const vector<Point3f> p3d, const vector<Point3f> p3d2,
Mat& R, const Mat& t){ 
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3> > MyBlock;
    typedef g2o::LinearSolverCSparse<MyBlock::PoseMatrixType> MySolver;
    g2o::OptimizationAlgorithmLevenberg* solver = 
        new g2o::OptimizationAlgorithmLevenberg( g2o::make_unique<MyBlock>(g2o::make_unique<MySolver>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    //添加顶点，只有一个顶点，不需要估计空间当中的点了
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    Eigen::Matrix3d R_mat;
    Eigen::Vector3d t_mat;
    R_mat << 
        R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), 
        R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), 
        R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);
    t_mat << 
        t.at<double>(0,0),
        t.at<double>(1,0),
        t.at<double>(2,0);
    pose->setId(0);
    pose->setEstimate(
        g2o::SE3Quat(
            Eigen::Matrix3d::Identity(),
            Eigen::Vector3d::Zero()
        )
    );
    optimizer.addVertex(pose);
    // 添加边
    int index = 1;
    vector<EdgeProjectXYZRGBDPoseOnly*> edges;
    for(int i=0;i<p3d.size();i++){
        EdgeProjectXYZRGBDPoseOnly* edge = new EdgeProjectXYZRGBDPoseOnly(
            Eigen::Vector3d(p3d2[i].x, p3d2[i].y, p3d2[i].z)
        );
        edge->setId(index);
        edge->setVertex(0,dynamic_cast<g2o::VertexSE3Expmap*>(pose));
        edge->setMeasurement(            
            Eigen::Vector3d(p3d[i].x, p3d[i].y, p3d[i].z));
        edge->setInformation(Eigen::Matrix3d::Identity()*1e4);
        optimizer.addEdge(edge);
        index++;
        edges.push_back(edge);
    }
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose( true );
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout<<"optimization costs time: "<<time_used.count()<<" seconds."<<endl;

    cout<<endl<<"after optimization:"<<endl;
    cout<<"T="<<endl<<Eigen::Isometry3d( pose->estimate() ).matrix()<<endl;
}

void pose_estimation_3d3d(const vector<Point3f>& pts1, const vector<Point3f>& pts2,
Mat& R, Mat& t){
    Point3f p1,p2;
    int N = pts1.size();
    for(int i=0;i<N;i++){
        p1+=pts1[i];
        p2+=pts2[i];
    }
    p1/=N;
    p2/=N;
    vector<Point3f> q1(N), q2(N);
    for(int i=0;i<N;i++){
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }
    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for(int i=0;i<N;i++){
        W+=Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * 
        Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }
    cout  << "W=" << W << endl;

    //SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    cout << "U=" << U << endl;
    cout << "V=" << V << endl;

    Eigen::Matrix3d R_ = U*(V.transpose());
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) -
     R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

    // 转化成cv mat
    R = (Mat_<double>(3,3) <<
        R_(0,0), R_(0,1), R_(0,2), 
        R_(1,0), R_(1,1), R_(1,2), 
        R_(2,0), R_(2,1), R_(2,2)
    );
    t = (Mat_<double>(3,1) << t_(0,0), t_(1,0), t_(2,0));
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
    vector<Point3f> p3d2;
    vector<Point2f> p2d;
    for(DMatch  m:matches){
        ushort d = d1.ptr<unsigned short> 
        (int ( k1[m.queryIdx].pt.y )) 
        [ int ( k1[m.queryIdx].pt.x ) ];
        ushort dist2 = d2.ptr<unsigned short> 
        (int ( k2[m.trainIdx].pt.y )) 
        [ int ( k2[m.trainIdx].pt.x ) ];
        if(d==0 || dist2==0) continue;
        float dd = d/5000.0;
        float dd2 = dist2/5000.0;
        Point2d p1 = pixel2cam(k1[m.queryIdx].pt,K);
        Point2d p2 = pixel2cam(k2[m.trainIdx].pt,K);
        p3d.push_back(Point3f(p1.x*dd, p1.y*dd, dd));
        p3d2.push_back(Point3f(p2.x*dd2, p2.y*dd2, dd2));
        p2d.push_back(k2[m.trainIdx].pt);
    }

    cout << "3d-2d pairs: " << p3d.size() << endl;

    Mat r,t;
    solvePnP(p3d,p2d,K,Mat(),r,t,false,cv::SOLVEPNP_EPNP);
    Mat R;
    Rodrigues(r,R);

    cout<<"R="<<endl<<R<<endl;
    cout<<"t="<<endl<<t<<endl;

    cout << "3d-3d pairs: " << p3d2.size() << endl;

    pose_estimation_3d3d(p3d,p3d2, R,t);
    cout<<"R="<<endl<<R<<endl;
    cout<<"t="<<endl<<t<<endl;

    g2o_icp(p3d,p3d2, R,t);
        // verify p1 = R*p2 + t
    for ( int i=0; i<5; i++ )
    {
        cout<<"p1 = "<<p3d[i]<<endl;
        cout<<"p2 = "<<p3d2[i]<<endl;
        cout<<"(R*p2+t) = "<<
            R * (Mat_<double>(3,1)<<p3d2[i].x, p3d2[i].y, p3d2[i].z) + t
            <<endl;
        cout<<endl;
    }
    return 0;
}
