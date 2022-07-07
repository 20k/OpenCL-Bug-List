#include "../common.hpp"

std::string kernel_src = R"(
__kernel void test_image(__read_only image2d_array_t img)
{
    sampler_t sam = CLK_NORMALIZED_COORDS_FALSE |
                            CLK_ADDRESS_NONE |
                            CLK_FILTER_NEAREST;

    float value = read_imagef(img, sam, (float4)(0.5f, 0.5f, 1.f, 0.f)).x;

    printf("Expected 2: Found: %f\n", value);
}
)";

#define DIM 128

cl_mem make_array(cl_context ctx)
{
    cl_image_format format;
    format.image_channel_order = CL_R;
    format.image_channel_data_type = CL_FLOAT;

    cl_image_desc desc = {0};
    desc.image_depth = 1;

    desc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
    desc.image_width = DIM;
    desc.image_height = DIM;
    desc.image_array_size = 2;

    cl_int err;
    cl_mem ret = clCreateImage(ctx, CL_MEM_READ_WRITE, &format, &desc, nullptr, &err);

    assert(err == CL_SUCCESS);

    return ret;
}

void fill_array(cl_command_queue cqueue, cl_mem mem)
{
    auto fill_slice = [&](size_t start_slice, size_t slices, cl_float with)
    {
        size_t origin[3] = {0, 0, start_slice};
        size_t region[3] = {DIM, DIM, slices};

        std::vector<cl_float> data;

        for(size_t i=0; i < DIM * DIM * slices; i++)
        {
            data.push_back(with);
        }

        clEnqueueWriteImage(cqueue, mem, CL_TRUE, origin, region, 0, 0, data.data(), 0, nullptr, nullptr);
    };

    ///filling individual levels of a texture array 1 by 1 does not work
    ///attempting to fill level 1, actually overwrites level 0
    #define REPRODUCE_BUG
    #ifdef REPRODUCE_BUG
    ///write the value 1.f, to texture level 0, fill 1 slice
    fill_slice(0, 1, 1.f);
    ///writ ethe value 2.f to texture level 1, fill 1 
    fill_slice(1, 1, 2.f);
    #else
    ///filling multiple texture levels simultaneously works fine
    fill_slice(0, 2, 2.f);
    #endif
}

int main()
{
    GLFWwindow* window = boot_opengl();

    auto [ctx, selected_device] = boot_opencl(true);

    std::vector<std::pair<cl_kernel, std::string>> kernels = build_kernels(ctx, selected_device, kernel_src);

    cl_command_queue cqueue = create_basic_command_queue(ctx, selected_device);

    cl_mem test_array = make_array(ctx);
    fill_array(cqueue, test_array);

    cl_kernel to_run = kernels[0].first;

    assert(kernels[0].second == "test_image");

    set_kernel_args(to_run, test_array);

    size_t offsets[1] = {0};
    size_t global[1] = {1};
    size_t local[1] = {1};

    cl_int result = clEnqueueNDRangeKernel(cqueue, to_run, 1, offsets, global, local, 0, nullptr, nullptr);

    assert(result == CL_SUCCESS);

    clFinish(cqueue);

    return 0;
}
