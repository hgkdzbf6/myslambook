#include <iostream> 
#include <ctime>
#include <iomanip>
#include <chrono>
#include <vector>
using namespace std;
#include<Eigen/Core>
#include<cmath>
#include "g2o/stuff/sampler.h"
#include "g2o/stuff/command_args.h"
#include<g2o/core/base_vertex.h>
#include<g2o/core/base_unary_edge.h>
#include<g2o/core/block_solver.h>
#include "g2o/core/solver.h"
#include<g2o/core/optimization_algorithm_levenberg.h>
#include<g2o/core/optimization_algorithm_gauss_newton.h>
#include<g2o/core/optimization_algorithm_dogleg.h>
#include<g2o/solvers/dense/linear_solver_dense.h>
#include <opencv2/core/core.hpp>

class CurveFittingVertex: public g2o::BaseVertex<3,Eigen::Vector3d>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CurveFittingVertex(){}
    virtual void setToOriginImpl(){
        _estimate << 0,0,0;
    }
    virtual void oplusImpl(const double* update){
        _estimate+=Eigen::Vector3d(update);
    }
    virtual bool read(istream& in){return false;}
    virtual bool write(ostream& out)const {return false;}
};
// unary 一元的
class CurveFittingEdge: public g2o::BaseUnaryEdge<1,Eigen::Vector2d,CurveFittingVertex>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CurveFittingEdge():BaseUnaryEdge(){}
    void computeError(){
        const CurveFittingVertex* v = static_cast<const CurveFittingVertex*>(vertex(0));
        const double& a = v->estimate()(0);
        const double& b = v->estimate()(1);
        const double& c = v->estimate()(2);
        const double& x = measurement()(0);
        double fval = exp(a*x*x+b*x+c);
        _error(0) = fval - measurement()(1);
    }
    virtual bool read(istream& in){return false;}
    virtual bool write(ostream& out)const {return false;}
};

int main(int argc, char const *argv[])
{
    double a=1.0,b=2.0,c=1.0;
    int N=100;
    double w_sigma = 1.0;
    cv::RNG rng;
    double abc[3] = {0,0,0};
    // std::vector<double> x_data, y_data;
    Eigen::Vector2d* points = new Eigen::Vector2d[N];
    cout << "generate data: " <<endl;
    for(int i=0;i<N;i++){
        double x = i/100.0;
        Eigen::Vector2d vec;
        vec << x,
            exp(a*x*x+b*x+c) + rng.gaussian(w_sigma);
        cout << vec(0) << "," << vec(1) << endl;
        points[i] = vec;
    }

    // construct graphic optimization
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<Eigen::Dynamic,Eigen::Dynamic>> MyBlock;
    typedef g2o::LinearSolverDense<MyBlock::PoseMatrixType> MySolver;
    // auto linearSolver = g2o::make_unique<MySolver>();
    // auto solver_ptr = g2o::make_unique<MyBlock>(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<MyBlock>(g2o::make_unique<MySolver>()));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    CurveFittingVertex* v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector3d(0,0,0));
    v->setId(0);
    optimizer.addVertex(v);

    for(int i=0;i<N;i++){
        CurveFittingEdge* edge = new CurveFittingEdge;
        edge->setVertex(0,v);
        edge->setInformation(Eigen::Matrix<double,1,1>::Identity()*1/(w_sigma*w_sigma));
        edge->setMeasurement(points[i]);
        optimizer.addEdge(edge);
    }

    cout << "start optimization: "<<endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout << "solve time cost = " << time_used.count() << endl;

    Eigen::Vector3d abc_estimate = v->estimate();
    cout << "estimated model: " <<abc_estimate.transpose() << endl;
    return 0;
}
