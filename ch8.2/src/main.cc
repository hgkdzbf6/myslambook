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

struct Measurement{
    Measurement (Eigen::Vector3d p, float g) : pos_world (p), grayscale(g) {}
    Eigen::Vector3d pos_world;
    float grayscale;
};

inline Eigen::Vector3d project2d3d(int x, int y, int d, float fx, float fy, float cx, float cy, float scale){
    float zz = float(d) / scale;
    float xx = zz* (x - cx) /fx;
    float yy = zz*(y-cy)/fy;
    return Eigen::Vector3d(xx,yy,zz);
}
inline Eigen::Vector2d project3d2d(float x, float y, float z, float fx, float fy, float cx, float cy){
    float u = fx*x/z+cx;
    float v = fy*y/z+cy;
    return Eigen::Vector2d(u,v);
}

class EdgeSE3ProjectDirect: public g2o::BaseUnaryEdge<1,double, g2o::VertexSE3Expmap> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeSE3ProjectDirect(){}

    EdgeSE3ProjectDirect ( Eigen::Vector3d point, float fx, float fy, float cx, float cy, cv::Mat* image )
        : x_world_ ( point ), fx_ ( fx ), fy_ ( fy ), cx_ ( cx ), cy_ ( cy ), image_ ( image )
    {}

    virtual void computeError(){
        const g2o::VertexSE3Expmap* v = static_cast<const g2o::VertexSE3Expmap*> (_vertices[0]);
        Eigen::Vector3d x_local = v->estimate().map(x_world_);
        float x = x_local[0]*fx_/x_local[2]+cx_;
        float y = x_local[1]*fy_/x_local[2]+cy_;

        if((x-4<0) ||(x+4)> image_->cols || (y-4)<0||(y+4)>image_->rows){
            _error(0,0) =0.0;
            this->setLevel(1);
        }else{
            _error(0,0) = getPixelValue(x,y) - _measurement;
        }
    }
    
    virtual void linearizeOplus(){
        if(level()==1){
            _jacobianOplusXi = Eigen::Matrix<double,1,6>::Zero();
            return;
        }
        g2o::VertexSE3Expmap* vtx = static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
        Eigen::Vector3d xyz_trans = vtx->estimate().map(x_world_);

        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double invz = 1.0/xyz_trans[2];
        double invz_2 = invz*invz;

        float u = x*fx_*invz+cx_;
        float v = y*fy_*invz+cy_;
        //雅克比矩阵
        Eigen::Matrix<double, 2,6> jacobian_uv_ksai;
        jacobian_uv_ksai(0,0) = -x*y*invz_2*fx_;
        jacobian_uv_ksai(0,1) = (1+(x*x*invz_2))*fx_;
        jacobian_uv_ksai(0,2) = -y*invz*fx_;
        jacobian_uv_ksai(0,3) = invz*fx_;
        jacobian_uv_ksai(0,4) = 0;
        jacobian_uv_ksai(0,5) = -x*invz_2*fx_;

        jacobian_uv_ksai(1,0) = - (1+y*y*invz_2)*fy_;
        jacobian_uv_ksai(1,1) = x*y*invz_2*fy_;
        jacobian_uv_ksai(1,2) = x*invz *fy_;
        jacobian_uv_ksai(1,3) = 0;
        jacobian_uv_ksai(1,4) = invz*fy_;
        jacobian_uv_ksai(1,5) = -y*invz_2*fy_;

        Eigen::Matrix<double, 1,2> jacobian_pixel_uv;
        jacobian_pixel_uv(0,0) = (getPixelValue(u+1,v)  - getPixelValue(u-1,v))/2;
        jacobian_pixel_uv(0,1) = (getPixelValue(u,v+1) - getPixelValue(u,v-1))/2;

        _jacobianOplusXi = jacobian_pixel_uv * jacobian_uv_ksai;
    }    
    // dummy read and write functions because we don't care...
    virtual bool read ( std::istream& in ) {}
    virtual bool write ( std::ostream& out ) const {}

protected:
    // 双线性插值方法来计算一个点的灰度值
    inline float getPixelValue( float x, float y){
        uchar* data = & image_ ->data[int(y)*image_->step + int(x)];
        float xx = x-floor(x);
        float yy = y-float(y);
        return float(
            (1-xx)*(1-yy)*data[0] + 
            xx*(1-yy)*data[1] +
            (1-xx) * yy * data[image_->step] + 
            xx*yy*data[image_->step+1]
        );
    }
public:
    Eigen::Vector3d x_world_;   // 3D point in world frame
    float cx_=0, cy_=0, fx_=0, fy_=0; // Camera intrinsics
    cv::Mat* image_=nullptr;    // reference image
};

bool poseEstimationDirect(const vector<Measurement> & measurements, cv::Mat* gray, Eigen::Matrix3f& K, Eigen::Isometry3d& Tcw){
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,1>> MyBlock;
    typedef g2o::LinearSolverCSparse<MyBlock::PoseMatrixType> MySolver;

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<MyBlock>(g2o::make_unique<MySolver>()));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setEstimate(g2o::SE3Quat(Tcw.rotation(), Tcw.translation()));
    pose->setId(0);
    optimizer.addVertex(pose);

    int id = 1;
    for( auto m : measurements){
        EdgeSE3ProjectDirect* edge = new EdgeSE3ProjectDirect(
            m.pos_world,
            K(0,0), K(1,1), K(0,2), K(1,2), gray
        );
        edge->setVertex(0, pose);
        edge->setMeasurement(m.grayscale);
        edge->setInformation( Eigen::Matrix<double,1,1>::Identity());
        edge->setId(id++);
        optimizer.addEdge(edge);
    }
    cout << "edges in graph: "<< optimizer.edges().size()<< endl;
    optimizer.initializeOptimization();
    optimizer.optimize(30);
    Tcw = pose->estimate();
}

int main(int argc, char const *argv[])
{
    string base_path = "../data/";
    string associate_file = base_path + "associate.txt";

    ifstream fin(associate_file);
    string rgb_file, depth_file, time_rgb, time_depth;
    cv::Mat color, depth, gray;
    vector<Measurement> measurements;
    // 相机内参
    float cx = 325.5;
    float cy = 253.5;
    float fx = 518.0;
    float fy = 519.0;
    float depth_scale = 1000.0;
    Eigen::Matrix3f K;
    K<<fx,0.f,cx,0.f,fy,cy,0.f,0.f,1.0f;
        
    Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity();

    cv::Mat prev_color;

    for(int index= 0; index<10;index++){
        cout << "**********loop " << index << " ***************" <<endl;
        fin>>time_rgb>>rgb_file>>time_depth>>depth_file;
        color = cv::imread ( base_path + rgb_file );
        depth = cv::imread ( base_path + depth_file, -1 );

        if ( color.data==nullptr || depth.data==nullptr )
            continue; 
        cv::cvtColor ( color, gray, cv::COLOR_BGR2GRAY );
        if ( index ==0 )
        {
            // 对第一帧提取FAST特征点
            vector<cv::KeyPoint> keypoints;
            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
            detector->detect ( color, keypoints );
            for ( auto kp:keypoints )
            {
                // 去掉邻近边缘处的点
                if ( kp.pt.x < 20 || kp.pt.y < 20 || ( kp.pt.x+20 ) >color.cols || ( kp.pt.y+20 ) >color.rows )
                    continue;
                ushort d = depth.ptr<ushort> ( cvRound ( kp.pt.y ) ) [ cvRound ( kp.pt.x ) ];
                if ( d==0 )
                    continue;
                Eigen::Vector3d p3d = project2d3d ( kp.pt.x, kp.pt.y, d, fx, fy, cx, cy, depth_scale );
                float grayscale = float ( gray.ptr<uchar> ( cvRound ( kp.pt.y ) ) [ cvRound ( kp.pt.x ) ] );
                measurements.push_back ( Measurement ( p3d, grayscale ) );
            }
            prev_color = color.clone();
            continue;
        }        
        // 使用直接法计算相机运动
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        poseEstimationDirect ( measurements, &gray, K, Tcw );
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
        cout<<"direct method costs time: "<<time_used.count() <<" seconds."<<endl;
        cout<<"Tcw="<<Tcw.matrix() <<endl;


        // plot the feature points
        cv::Mat img_show ( color.rows*2, color.cols, CV_8UC3 );
        prev_color.copyTo ( img_show ( cv::Rect ( 0,0,color.cols, color.rows ) ) );
        color.copyTo ( img_show ( cv::Rect ( 0,color.rows,color.cols, color.rows ) ) );
        for ( Measurement m:measurements )
        {
            if ( rand() > RAND_MAX/5 )
                continue;
            Eigen::Vector3d p = m.pos_world;
            Eigen::Vector2d pixel_prev = project3d2d ( p ( 0,0 ), p ( 1,0 ), p ( 2,0 ), fx, fy, cx, cy );
            Eigen::Vector3d p2 = Tcw*m.pos_world;
            Eigen::Vector2d pixel_now = project3d2d ( p2 ( 0,0 ), p2 ( 1,0 ), p2 ( 2,0 ), fx, fy, cx, cy );
            if ( pixel_now(0,0)<0 || pixel_now(0,0)>=color.cols || pixel_now(1,0)<0 || pixel_now(1,0)>=color.rows )
                continue;

            float b = 255*float ( rand() ) /RAND_MAX;
            float g = 255*float ( rand() ) /RAND_MAX;
            float r = 255*float ( rand() ) /RAND_MAX;
            cv::circle ( img_show, cv::Point2d ( pixel_prev ( 0,0 ), pixel_prev ( 1,0 ) ), 8, cv::Scalar ( b,g,r ), 2 );
            cv::circle ( img_show, cv::Point2d ( pixel_now ( 0,0 ), pixel_now ( 1,0 ) +color.rows ), 8, cv::Scalar ( b,g,r ), 2 );
            cv::line ( img_show, cv::Point2d ( pixel_prev ( 0,0 ), pixel_prev ( 1,0 ) ), cv::Point2d ( pixel_now ( 0,0 ), pixel_now ( 1,0 ) +color.rows ), cv::Scalar ( b,g,r ), 1 );
        }
        cv::imshow ( "result", img_show );
        cv::waitKey ( 0 );

    }
    return 0;
}
