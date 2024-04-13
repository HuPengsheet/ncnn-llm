#include<iostream>
#include <cfloat>
#include"llm.h"
#include"token.h"
#include"net.h"
using namespace std;
Tokenizer tokenizer;
    static void ErrorInFastLLM(const std::string &error) {
        printf("FastLLM Error: %s\n", error.c_str());
        throw error;
    }

    struct FileBuffer {
        FILE *f;

        FileBuffer (const std::string &fileName) {
            f = fopen(fileName.c_str(), "rb");
        }

        int ReadInt() {
            int v;
            if (fread(&v, 1, 4, f) != 4) {
                ErrorInFastLLM("FileBuffer.ReadInt error.\n");
            };
            return v;
        }

        float ReadFloat() {
            float v;
            if (fread(&v, 1, 4, f) != 4) {
                ErrorInFastLLM("FileBuffer.ReadFloat error.\n");
            };
            return v;
        }

        std::string ReadString() {
            int len = ReadInt();
            std::string ret = "";
            char *v = new char[len + 5];
            v[len] = 0;
            if (fread(v, 1, len, f) != len) {
                ErrorInFastLLM("FileBuffer.ReadString error.\n");
            }
            return v;
        }

        void ReadBytes(uint8_t *buffer, uint64_t bytes) {
            if (fread(buffer, 1, bytes, f) != bytes) {
                ErrorInFastLLM("FileBuffer.ReadBytes error.\n");
            }
        }

        ~FileBuffer() {
            fclose(f);
        }
    };

vector<float> get_token()
{
    
    tokenizer.type = Tokenizer::TokenizerType::GLM;
    std::string  model_path = "/home/hp/code/llm/tokenizer/workspace/token_model.bin";
    FileBuffer buffer (model_path);
    bool useScore = 1;
    int vocabLen = buffer.ReadInt();
    for (int i = 0; i < vocabLen; i++) {
        int len = buffer.ReadInt();
        std::string x = "";
        for (int j = 0; j < len; j++) {
            x += buffer.ReadInt();
        }
        int id = buffer.ReadInt();
        cout<<x<<endl;
        float score = useScore ? buffer.ReadFloat() : -i;
        tokenizer.Insert(x, id, score);
    }

    auto res = tokenizer.Encode("最近我在办公室坐久了会感到头晕，请问这是什么原因?有什么缓解办法吗？");
    res.push_back(1);
    return res;
    // bool hasSpecialTokens = 1;
    // if (hasSpecialTokens) {
    //     std::map <std::string, int> specialTokens;
    //     int specialTokenLen = buffer.ReadInt();
    //     for (int i = 0; i < specialTokenLen; i++) {
    //         std::string token = buffer.ReadString();
    //         int id = tokenizer.stringToTokenDict[token];
    //         specialTokens[token] = id;
    //     }
    //     tokenizer.SetSpecialTokens(specialTokens);
    // }
}

//返回的是一对pair pair里 dim=1024 end=head=8
std::pair<std::vector<float>, std::vector<float>> precompute_freqs_cis(int dim, int end, float theta = 10000.0) {
    std::vector<float> freqs;
    freqs.reserve(dim / 2);
    
    for (int i = 0; i < dim; i += 2) {
        freqs.push_back(1.0 / std::pow(theta, static_cast<float>(i) / dim));
    }
    
    std::vector<float> freqs_cos(end * (dim / 2));
    std::vector<float> freqs_sin(end * (dim / 2));
    
    for (int i = 0; i < end; ++i) {
        for (int j = 0; j < dim / 2; ++j) {
            float t = static_cast<float>(i);
            freqs_cos[i * (dim / 2) + j] = std::cos(t * freqs[j]);
            freqs_sin[i * (dim / 2) + j] = std::sin(t * freqs[j]);
        }
    }
    
    return std::make_pair(freqs_cos, freqs_sin);
}

/*
v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
logits[logits < v[:, [-1]]] = -float('Inf')
*/
int topk(ncnn::Mat& input,int max_k){
    vector<float> nums;
    for(int i=0;i<input.w;i++){
        float* ptr=(float *)input;
        nums.push_back(ptr[i]);
    }

    auto maxElement = std::max_element(nums.begin(), nums.end());
    int maxIndex = std::distance(nums.begin(), maxElement);
    // cout<<"index="<<maxIndex<<endl;
    // cout<<"max="<<nums[maxIndex]<<endl;

    std::sort(nums.begin(), nums.end(), [](int a, int b) {
        return a > b;
    });

    float thre=nums.back();
    ncnn::Mat res(input.w);
    for(int i =0;i<input.w;i++){
        float* ptr=(float *)input;
        float* res_ptr=(float *)res;
        if(ptr[i]<thre)
            res_ptr[i] = -FLT_MAX;
        else
            res_ptr[i] = ptr[i];

    }

        //attn_weight = torch.softmax(attn_weight, dim=-1)
    ncnn::Option my_opt;
    my_opt.num_threads = 8;
    my_opt.use_fp16_storage = false;
    my_opt.use_packing_layout = false;
    ncnn::Layer* op = ncnn::create_layer("Softmax");
    ncnn::ParamDict pd;
    pd.set(0, 2);
    pd.set(1, 1);
    op->load_param(pd);
    op->create_pipeline(my_opt);
    op->forward_inplace(res,my_opt);
    op->destroy_pipeline(my_opt);
    delete op;

    int next_index=0;
    for(int i=0;i<input.w;i++){
        float* ptr=(float *)res;
        if(ptr[i]>0.1){
            next_index=i;
            break;
        }
    }
    
    cout<<next_index<<endl;

    return next_index;
}

int llm(vector<float> tokens){
    int seq_len = tokens.size();
    ncnn::Mat tokens_mat;
    tokens_mat.create(seq_len);
    for(int i=0;i<seq_len;i++){
        int* ptr = (int*)tokens_mat;
        ptr[i] = (int)tokens[i];
    }
    //pretty_print(tokens_mat);
    auto embding = get_embding(tokens_mat);
    auto cos_sin = precompute_freqs_cis(1024/8,1024);
    vector<float> freqs_cos = cos_sin.first;
    vector<float> freqs_sin = cos_sin.second;
    
    ncnn::Mat cos_mat(64,seq_len);
    ncnn::Mat sin_mat(64,seq_len);
    for(int i=0;i<seq_len;i++){
        for(int j=0;j<64;j++){
           float* cos_ptr  = (float*)cos_mat;
           float* sin_ptr  = (float*)sin_mat;
           cos_ptr[i*64+j] = freqs_cos[i*64+j];
           sin_ptr[i*64+j] = freqs_sin[i*64+j];
        }
    }
    auto logits = forward_llm(embding,cos_mat,sin_mat);
    
    auto result = linear(logits);
    auto next_index = topk(result,100);
    return next_index;
}

std::vector<int> convertFloatToIntVector(const std::vector<float> floatVector) {  

    std::vector<int> intVector(floatVector.size());  

    std::transform(floatVector.begin(), floatVector.end(), intVector.begin(),  

                    [](float f) { return static_cast<int>(f); });  

    return intVector;  

} 
int main(){

    vector<float> tokens= get_token();
    
    
    // for(int i=0;i<100;i++){
    //     int index = llm(tokens);
    //     if(index==2) break;
    //     else tokens.push_back((float)index);
    // }
    // for(auto token:tokens){
    //     cout<<token<<',';
    // }
    // string answer = tokenizer.Decode(convertFloatToIntVector(tokens));
    // cout<<answer<<' ';
    // return 0;
}