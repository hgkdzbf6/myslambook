#pragma once
#include <common.h>

namespace zbf{
    class Config{
    public:
        typedef std::shared_ptr<Config> Ptr;
    private:
        static Ptr config_;
        cv::FileStorage file_;
        Config(){}
    public:
        ~Config();
        static void setParameterFile(const std::string& filename);
        template<typename T>
        static T get(const std::string& key){
            return T(Config::config_->file_[key]);
        }
    };
}