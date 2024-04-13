#include<iostream>
#include "net.h"
#include "llm.h"
#include <math.h>
#include <stdio.h>

using namespace std;


void pretty_print(const ncnn::Mat& m)
{
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int z=0; z<m.d; z++)
        {
            for (int y=0; y<m.h; y++)
            {
                for (int x=0; x<m.w; x++)
                {
                    printf("%f ", ptr[x]);
                }
                ptr += m.w;
                printf("\n");
            }
            printf("\n");
        }
        printf("------------------------\n");
    }
    printf("dims=%d w=%d h=%d d=%d c=%d \n",m.dims,m.w,m.h,m.d,m.c);
}
//0=1024 1=64793 2=0 3=66348032

ncnn::Mat get_embding(ncnn::Mat input){
    ncnn::Net embding;
    embding.opt.use_fp16_packed = false;
    embding.opt.use_fp16_storage = false;
    embding.opt.use_fp16_arithmetic = false;
    embding.opt.use_vulkan_compute = false;
    if (embding.load_param("/home/hp/code/llm/ncnn-20240410/model_file/embding.ncnn.param"))
        exit(-1);
    if (embding.load_model("/home/hp/code/llm/ncnn-20240410/model_file/embding.ncnn.bin"))
        exit(-1);
    ncnn::Extractor ex = embding.create_extractor();


    ex.input("in0", input);
    //pretty_print(input);
    ncnn::Mat out;
    ex.extract("out0", out);

    //pretty_print(out);
    return out;
}


ncnn::Mat forward_llm(ncnn::Mat embding,ncnn::Mat cos_mat,ncnn::Mat sin_mat){
    int dims = embding.w;
    int seq_len = embding.h;
    ncnn::Net scale;
    scale.opt.use_fp16_packed = false;

    scale.opt.use_fp16_storage = false;

    scale.opt.use_fp16_arithmetic = false;
    scale.opt.use_vulkan_compute = false;
    if (scale.load_param("/home/hp/code/llm/ncnn-20240410/model_file/all.ncnn.param"))
        exit(-1);
    if (scale.load_model("/home/hp/code/llm/ncnn-20240410/model_file/all.ncnn.bin"))
        exit(-1);
    ncnn::Extractor ex = scale.create_extractor();
    // ncnn::Mat input1(1024,20);
    // input1.fill(0.1f);

    // ncnn::Mat input2(64,20);
    // input2.fill(0.2f);

    // ncnn::Mat input3(64,20);
    // input3.fill(0.3f);

    ex.input("in0", embding);
    ex.input("in1", cos_mat);
    ex.input("in2", sin_mat);

    ncnn::Mat out;
    ex.extract("out0", out);

    //取最后一个维度的Mat
    ncnn::Mat reslut(dims,1);
    for(int i=0;i<dims;i++){
        float* ptr = (float*)reslut;
        float* out_ptr = (float*)out;
        ptr[i]=out_ptr[(seq_len-1)*dims+i]; 
    }
    return reslut;
    // pretty_print(out);
    // cout<<out.w<<' '<<out.h<<' '<<out.c<<endl;
    // cout<<out.d<<endl;
    // cout<<out.dims<<endl;
}

//inpiut 1 1024
ncnn::Mat linear(ncnn::Mat input){
    ncnn::Net linear;
    linear.opt.use_fp16_packed = false;

    linear.opt.use_fp16_storage = false;

    linear.opt.use_fp16_arithmetic = false;
    linear.opt.use_vulkan_compute = false;
    if (linear.load_param("/home/hp/code/llm/ncnn-20240410/model_file/line.ncnn.param"))
        exit(-1);
    if (linear.load_model("/home/hp/code/llm/ncnn-20240410/model_file/line.ncnn.bin"))
        exit(-1);
    ncnn::Extractor ex = linear.create_extractor();
    // ncnn::Mat input1(1024,20);
    // input1.fill(0.1f);

    // ncnn::Mat input2(64,20);
    // input2.fill(0.2f);

    // ncnn::Mat input3(64,20);
    // input3.fill(0.3f);

    ex.input("in0", input);
    ncnn::Mat out;
    ex.extract("out0", out);


    return out;
    // pretty_print(out);
    // cout<<out.w<<' '<<out.h<<' '<<out.c<<endl;
    // cout<<out.d<<endl;
    // cout<<out.dims<<endl;
}