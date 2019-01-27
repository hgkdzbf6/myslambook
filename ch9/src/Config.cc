#include "Config.h"

zbf::Config::Ptr zbf::Config::config_ = nullptr;
namespace zbf{
    
    void zbf::Config::setParameterFile( const std::string& filename) {
        if(config_ == nullptr)
            config_ = zbf::Config::Ptr (new Config);
        config_->file_ = cv::FileStorage(filename.c_str(), cv::FileStorage::READ);
        if(config_->file_.isOpened() == false ){
            // 报错
            config_->file_.release();

        }
    }

    zbf::Config::~Config(){
        if(file_.isOpened())file_.release();
    }
}