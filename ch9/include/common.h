#pragma once

#include <iostream> 
#include <list>
#include <ctime>
#include <iomanip>
#include <chrono>
#include <memory>
#include <map>
#include <unordered_map>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>

#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/viz.hpp>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include <boost/timer.hpp>

using namespace std;
using namespace Eigen;
using namespace Sophus;
using namespace cv;
using namespace g2o;