#include <common.h>

#include<Camera.h>


namespace zbf{
    class EdgeProjectXYZ2UVPoseOnly : public g2o::BaseUnaryEdge<2,Eigen::Vector2d, g2o::VertexSE3Expmap>{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        virtual void computeError();
        virtual void linearizeOplus();

        virtual bool read(std::istream& in){}
        virtual bool write(std::ostream& out) const{}

        Vector3d point_;
        Camera* camera_;
    };
}