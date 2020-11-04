// main.cpp
// Dan Wolf

#include "Configuration.h"
#include "Exceptions.h"
#include "DeviceInfo.cuh"
#include "Calculator.cuh"
#include "HeatMap.h"

int main(int argc, char **argv) {
    config::Configuration conf(config::getArgs(argc, argv));
    switch (conf.command) {
        case config::CommandType::Help:
            config::Configuration::printHelp();
            break;
        case config::CommandType::Info:
            DeviceInfo::print();
            break;
        case config::CommandType::TimePoint: {
            Calculator calc(conf);
            auto answer = calc.exec();
            cout << answer;
            break;
        }
        case config::CommandType::Heat: {
            // always use CPU for maps
            conf.device = config::DeviceType::CPU;
            Calculator calc(conf);
            auto snapshot = calc.snapshotAt(conf.time);
            HeatMap map(snapshot, conf.normalize);
            map.generate();
            break;
        }
        case config::CommandType::Graph: {
            // always use CPU for graphing
            conf.device = config::DeviceType::CPU;
            Calculator calc(conf);
            calc.exec();
            break;
        }
    }

    return 0;
}
