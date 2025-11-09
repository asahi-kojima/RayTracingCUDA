#pragma once
#include <stdio.h>
#include <vector>
#include <string>
#include <cassert>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifndef _LINUX
#define _USE_MATH_DEFINES
#endif
#include <math.h>

#include "typeinfo.h"

#define DEBUG 1

#define SYSTEM_DEBUG 1

struct Result
{
	bool isSuccess;
	std::string message;

	Result(bool isSuccess = false) : isSuccess{ isSuccess }, message{ "" } {}
	Result(bool isSuccess, const std::string& message) : isSuccess(isSuccess), message(message){}

	operator bool() const { return isSuccess; }
};