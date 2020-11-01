#include "Configuration.h"

int main(int argc, char **argv) {
    config::Configuration conf(config::getArgs(argc, argv));
    switch (conf.command) {
        case config::CommandType::Help:
            config::Configuration::printHelp();
            break;
        case config::CommandType::Info:
            break;
        case config::CommandType::TimePoint:
            break;
        case config::CommandType::Graph:
            break;
    }

    return 0;
}
