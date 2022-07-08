#pragma once

#include <iostream>

#include <vector>
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <assert.h>
#include <string>
#include <GLFW/glfw3.h>
#include <windows.h>

inline
GLFWwindow* boot_opengl()
{
    assert(glfwInit());

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(640, 480, "dummy window", nullptr, nullptr);

    assert(window);

    glfwMakeContextCurrent(window);

    assert(glewInit() == GLEW_OK);

    return window;
}

inline 
void print_supported_extensions(cl_device_id did)
{
    size_t size;
    clGetDeviceInfo(did, CL_DEVICE_EXTENSIONS, 0, nullptr, &size);

    std::string ext;
    ext.resize(size + 1);

    clGetDeviceInfo(did, CL_DEVICE_EXTENSIONS, size, &ext[0], nullptr);

    std::cout << ext << std::endl;
}

inline
std::pair<cl_context, cl_device_id> boot_opencl(bool use_opengl)
{
    std::vector<cl_platform_id> clPlatformIDs;

    cl_uint num_platforms = 0;

    clGetPlatformIDs(0, nullptr, &num_platforms);

    assert(num_platforms > 0);

    clPlatformIDs.resize(num_platforms);

    clGetPlatformIDs(num_platforms, &clPlatformIDs[0], nullptr);

    cl_platform_id platform = clPlatformIDs[0];

    cl_uint num_devices = 0;
    cl_device_id devices[100] = {};

    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, devices, &num_devices);

    cl_device_id selected_device = devices[0];

    //print_supported_extensions(selected_device);

    cl_context ctx;

    if(use_opengl)
    {
        cl_context_properties props[] =
        {
            CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
            CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
            CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
            0
        };

        cl_int ctx_err;
        ctx = clCreateContext(props, 1, &selected_device, nullptr, nullptr, &ctx_err);
    
        assert(ctx_err == CL_SUCCESS);
    }
    else
    {
        cl_context_properties props[] =
        {
            0
        };

        cl_int ctx_err;
        ctx = clCreateContext(props, 1, &selected_device, nullptr, nullptr, &ctx_err);

        assert(ctx_err == CL_SUCCESS);
    }

    return {ctx, selected_device};
}

inline
std::vector<std::pair<cl_kernel, std::string>> build_kernels(cl_context ctx, cl_device_id selected_device, const std::string& str)
{
    const char* cstr = str.c_str();

    cl_program prog = clCreateProgramWithSource(ctx, 1, &cstr, nullptr, nullptr);

    std::string build_options = "-cl-std=CL2.0";

    cl_int build_status = clBuildProgram(prog, 1, &selected_device, build_options.c_str(), nullptr, nullptr);

    if(build_status != CL_SUCCESS)
    {
        std::cout << "Build Error: " << build_status << std::endl;

        cl_build_status bstatus;
        clGetProgramBuildInfo(prog, selected_device, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &bstatus, nullptr);

        std::cout << "Build Status: " << bstatus << std::endl;

        assert(bstatus == CL_BUILD_ERROR);

        std::string log;
        size_t log_size;

        clGetProgramBuildInfo(prog, selected_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

        log.resize(log_size + 1);

        clGetProgramBuildInfo(prog, selected_device, CL_PROGRAM_BUILD_LOG, log.size(), &log[0], nullptr);

        std::cout << log << std::endl;

        throw std::runtime_error("Failed to build");
    }

    cl_uint num_kernels = 0;
    cl_int err = clCreateKernelsInProgram(prog, 0, nullptr, &num_kernels);

    assert(err == CL_SUCCESS);

    std::vector<cl_kernel> cl_kernels;
    cl_kernels.resize(num_kernels + 1);

    clCreateKernelsInProgram(prog, num_kernels, &cl_kernels[0], nullptr);

    cl_kernels.resize(num_kernels);

    cl_queue_properties regular_props[] = {0};

    cl_command_queue cqueue = clCreateCommandQueueWithProperties(ctx, selected_device, regular_props, nullptr);

    cl_kernel to_run = nullptr;

    std::vector<std::pair<cl_kernel, std::string>> with_names;

    for(auto& k : cl_kernels)
    {
        std::string name;

        size_t size;
        clGetKernelInfo(k, CL_KERNEL_FUNCTION_NAME, 0, nullptr, &size);

        name.resize(size + 1);

        clGetKernelInfo(k, CL_KERNEL_FUNCTION_NAME, size, &name[0], nullptr);

        name.resize(strlen(name.c_str()));

        with_names.push_back({k, name});
    }

    return with_names;
}

template<typename... T>
inline
void set_kernel_args(cl_kernel to_set, T&&... args)
{
    int idx = 0;

    (clSetKernelArg(to_set, idx++, sizeof(args), &args), ...);
}

inline
cl_command_queue create_basic_command_queue(cl_context ctx, cl_device_id selected_device)
{
    cl_queue_properties regular_props[] = {0};

    return clCreateCommandQueueWithProperties(ctx, selected_device, regular_props, nullptr);
}