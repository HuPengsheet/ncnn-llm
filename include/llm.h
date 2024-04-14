#ifndef LLM_H
#define LLM_H
#include "net.h"
void pretty_print(const ncnn::Mat& m);
ncnn::Mat get_embding(ncnn::Mat input);
ncnn::Mat forward_llm(ncnn::Mat embding,ncnn::Mat cos_mat,ncnn::Mat sin_mat);
ncnn::Mat linear(ncnn::Mat input);
#endif