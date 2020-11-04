// Exceptions.cpp
// Dan Wolf

#ifndef HOTHOTHOT_EXCEPTIONS_H
#define HOTHOTHOT_EXCEPTIONS_H

#include <iostream>
#include <exception>
using namespace std;

class ConfigurationException: public exception {
private:
    const char* Message;

public:
    explicit ConfigurationException(const char* msg);

    ~ConfigurationException() override;

    const char* what() const noexcept override;
};

class NotImplementedException : public std::logic_error
{
public:
    NotImplementedException() : std::logic_error("Function not yet implemented") { };
};

#endif //HOTHOTHOT_EXCEPTIONS_H
