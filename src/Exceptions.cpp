// Exceptions.cpp
// Dan Wolf

#include "Exceptions.h"

ConfigurationException::ConfigurationException(const char* msg) {
    this->Message = msg;
}

const char *ConfigurationException::what() const noexcept {
    return this->Message;
}

ConfigurationException::~ConfigurationException() = default;

