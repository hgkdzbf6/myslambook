#include <iostream> 
#include <ctime>
#include <iomanip>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
using namespace std;
using namespace cv;

#define MATRIX_SIZE 50

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
    return 0;
}
