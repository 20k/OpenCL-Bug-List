#include "../common.hpp"
 
#define REPRODUCE_BUG

std::string kernel_src = R"(
#pragma OPENCL EXTENSION cl_khr_mipmap_image : enable

__kernel void test_image(__read_only image2d_t img)
{
    sampler_t sam = CLK_NORMALIZED_COORDS_FALSE |
                            CLK_ADDRESS_NONE |
                            CLK_FILTER_NEAREST;

    float value = read_imagef(img, sam, (float2)(0.5f, 0.5f)).x;

    printf("Expected 1: Found: %f\n", value);
}
)";

GLuint make_texture()
{
    GLuint handle;

    glGenTextures(1, &handle);
    glBindTexture(GL_TEXTURE_2D, handle);

    uint32_t* cols = new uint32_t[1024 * 1024];

    for(int i=0; i < 1024 * 1024; i++)
    {
        cols[i] = 0xFFFFFFFF;
    }

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1024, 1024, 0, GL_RGBA, GL_UNSIGNED_BYTE, cols);

    #ifdef REPRODUCE_BUG
    glGenerateMipmap(GL_TEXTURE_2D);
    #endif // REPRODUCE_BUG

    return handle;
}

int main()
{
    GLFWwindow* window = boot_opengl();

    auto [ctx, selected_device] = boot_opencl(true);

    GLuint opengl_texture = make_texture();

    glFinish();

    std::vector<std::pair<cl_kernel, std::string>> kernels = build_kernels(ctx, selected_device, kernel_src);

    cl_command_queue cqueue = create_basic_command_queue(ctx, selected_device);

    cl_int gl_err;
    cl_mem opencl_gl_texture = clCreateFromGLTexture(ctx, CL_MEM_READ_WRITE, GL_TEXTURE_2D, 0, opengl_texture, &gl_err);

    assert(gl_err == CL_SUCCESS);

    assert(clEnqueueAcquireGLObjects(cqueue, 1, &opencl_gl_texture, 0, nullptr, nullptr) == CL_SUCCESS);

    cl_kernel to_run = kernels[0].first;

    assert(kernels[0].second == "test_image");

    set_kernel_args(to_run, opencl_gl_texture);

    size_t offsets[1] = {0};
    size_t global[1] = {1};
    size_t local[1] = {1};

    cl_int result = clEnqueueNDRangeKernel(cqueue, to_run, 1, offsets, global, local, 0, nullptr, nullptr);

    assert(result == CL_SUCCESS);

    assert(clEnqueueReleaseGLObjects(cqueue, 1, &opencl_gl_texture, 0, nullptr, nullptr) == CL_SUCCESS);

    clFinish(cqueue);

    return 0;
}
