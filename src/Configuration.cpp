//
// Created by dan on 2020-10-27.
//

#include <locale>
#include "Configuration.h"

namespace config {
    // convert arguments to a vector of string for ease of processing
    std::vector<std::string> getArgs(int argc, char **argv) {
        std::vector<std::string> args;
        // first element is the command name
        // argv[argc] is NULL, so iterate till *pargv is NULL
        for (char **pargv = argv + 1; *pargv != argv[argc]; pargv++) {
            auto arg = std::string(*pargv);
            for (auto &c: arg) c = toupper(c);
            args.push_back(arg);
        }

        return args;
    }

    Configuration::Configuration(const std::vector<std::string> &args) {
        this->count.argc = args.size();
        // start counts of standalone parameters and pair parameters at 0
        this->count.pairs = this->count.standalone = 0;
        // initialize default values
        this->command = CommandType::Help;
        this->dimensions = Dimension::One;
        this->device = DeviceType::CPU;
        this->time = 0;
        this->location = 0;
        this->ambientTemp = 23;
        this->sourceTemp = 100;
        this->slices = 2500;

        for (const auto &arg : args) {
            auto delimPos = arg.find('=');
            if (delimPos == std::string::npos) {
                this->count.standalone += 1;
                this->command = Configuration::selectCommand(arg);
                continue;
            }
            this->count.pairs += 1;
            auto key = arg.substr(0, delimPos);
            auto valuePos = delimPos + 1;
            auto value = arg.substr(valuePos);
            // location time slices ambientTemp sourceTemp device
            if (key == "DIMENSIONS" || key == "DIM") {
                if (value == "TWO" || value == "2") this->dimensions = Dimension::Two;
                if (value == "THREE" || value == "3") this->dimensions = Dimension::Three;
            }
            if (key == "LOCATION" || key == "L") {
                this->location = std::stof(value);
            }
            if (key == "DEVICE" || key == "DEV") {
                if (value == "CUDA" || value == "GPU") this->device = DeviceType::CUDA_GPU;
            }
            if (key == "TIME" || key == "T") {
                this->time = std::stoi(value);
            }
            if (key == "AMBIENTTEMP" || key == "AMBIENT") {
                this->ambientTemp = std::stof(value);
            }
            if (key == "SOURCETEMP" || key == "SOURCE") {
                this->sourceTemp = std::stof(value);
            }
            if (key == "SLICES") {
                this->slices = std::stoi(value);
            }
        }
    }

    CommandType Configuration::selectCommand(const std::string& cmdString) {
        if (cmdString == "INFO") {
            return CommandType::Info;
        } else if (cmdString == "TIMEPOINT") {
            return CommandType::TimePoint;
        } else if (cmdString == "GRAPH") {
            return CommandType::Graph;
        }
        return CommandType::Help;
    }

    void Configuration::printHelp() {
        std::printf("Usage:\n\thothothot [command] arg=value...\n\n");
        std::printf("\tCommands:\n\n");
        std::printf("\t\tInfo: displays GPU / CUDA info\n");
        std::printf("\t\tTimePoint: calculates the temperature for a given time and location\n");
        std::printf("\t\t\tDimensions (DIM): 1, 2, or 3 (default 1)\n");
        std::printf("\t\t\tLocation (L): location to measure (default 0)\n");
        std::printf("\t\t\tDevice (DEV): set to CPU or GPU (default GPU)\n");
        std::printf("\t\t\tTime (T): time to measure (default 0)\n");
        std::printf("\t\t\tAmbientTemp (AMBIENT): ambient temperature (default 23)\n");
        std::printf("\t\t\tSourceTemp (SOURCE): temperature of heat source (default 100)\n");
        std::printf("\t\t\tSlices: the number of slices used (default 2500)\n");
        std::printf("\t\tGraph: displays GPU / CUDA info\n");
        std::printf("\t\tHelp: displays this message\n\n");
        std::printf("\tExample:\n\n");
        std::printf("\t\thothothot TimePoint L=.7 T=10000000\n");
    }

    Configuration::~Configuration() = default;
}
