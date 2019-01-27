#include <iostream> 
#include <ctime>
#include <iomanip>

using namespace std;

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <pangolin/pangolin.h>

#include "sophus/so3.hpp"
#include "sophus/se3.hpp"

#define MATRIX_SIZE 50

int main(int argc, char const *argv[])
{
    Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d(0,0,1)).toRotationMatrix();
    Sophus::SO3d so3_R(R);
    Sophus::SO3d so3_v = Sophus::SO3d::rotZ(M_PI/2);
    Eigen::Quaterniond q(R);
    Sophus::SO3d so3_q(q);
    
    cout << "SO(3) from matrix: " << so3_R.log().transpose() << endl;
    cout << "SO(3) from vector: " << so3_v.log().transpose()  << endl;
    cout << "SO(3) from quaternion: " << so3_q.log().transpose()  << endl;

    Eigen::Vector3d t(1,0,0);
    Sophus::SE3d se3_rt(R,t);
    Sophus::SE3d se3_qt(q,t);
    cout << se3_rt.log() << endl;
    cout << se3_qt.log() << endl;
    Eigen::Matrix<double,6,1> se3 =se3_qt.log();
    cout << Sophus::SE3d::hat(se3) << endl;
    cout << Sophus::SE3d::vee(Sophus::SE3d::hat(se3)) << endl;
    return 0;
}
