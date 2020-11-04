// Configuration.h
// Dan Wolf

#ifndef HOTHOTHOT_CONFIGURATION_H
#define HOTHOTHOT_CONFIGURATION_H

#include <string>
#include <vector>

namespace config {
    std::vector<std::string> getArgs(int argc, char **argv);

    struct CountInfo {
        int argc;
        int standalone;
        int pairs;
    };

    enum class CommandType {
        Info, TimePoint, Graph, Heat, Help
    };

    enum class DeviceType {
        CPU, CUDA_GPU
    };

    enum class Dimension {
        One, Two, Three
    };

    class Configuration {
    public:
        CountInfo count{};
        CommandType command;
        Dimension dimensions;
        DeviceType device;
        int time;
        float location;
        float ambientTemp;
        float sourceTemp;
        int slices;
        bool normalize;

        explicit Configuration(const std::vector<std::string> &args);

        ~Configuration();

        static void printHelp();

    private:
        static CommandType selectCommand(const std::string &cmdString);
    };
}
#endif //HOTHOTHOT_CONFIGURATION_H

