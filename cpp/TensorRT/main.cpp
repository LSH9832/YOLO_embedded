//
// Created by lsh on 2022/3/21.
//

#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <string.h>
#include "curl/curl.h"
#include "curl/easy.h"
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"

#define CHECK(status) \
    do\
    {\
        int ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define DEVICE 0  // GPU id
#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.25

using namespace nvinfer1;
using namespace std;

// stuff we know about the network and the input/output blobs
static const int NUM_CLASSES = 80;
const char* INPUT_BLOB_NAME = "input_0";
const char* OUTPUT_BLOB_NAME = "output_0";
static Logger gLogger;

cv::Mat static_resize(cv::Mat& img, int input_size) {
    float r = std::min(input_size / (img.cols*1.0), input_size / (img.rows*1.0));
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat out(input_size, input_size, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

static void generate_grids_and_stride(std::vector<int>& strides, std::vector<GridAndStride>& grid_strides, int img_size)
{
    for (auto stride : strides)
    {
        int num_grid_y = img_size / stride;
        int num_grid_x = img_size / stride;
        for (int g1 = 0; g1 < num_grid_y; g1++)
            for (int g0 = 0; g0 < num_grid_x; g0++)
                grid_strides.push_back((GridAndStride){g0, g1, stride});
    }
}

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            std::swap(faceobjects[i], faceobjects[j]);
            i++;
            j--;
        }
    }

    if (left < j) qsort_descent_inplace(faceobjects, left, j);
    if (i < right) qsort_descent_inplace(faceobjects, i, right);

}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
        areas[i] = faceobjects[i].rect.area();

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static void generate_yolox_proposals(std::vector<GridAndStride> grid_strides, float* feat_blob, float prob_threshold, std::vector<Object>& objects, int num_class)
{

    const int num_anchors = grid_strides.size();

    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;

        const int basic_pos = anchor_idx * (num_class + 5);

        // yolox/models/yolo_head.py decode logic
        float x_center = (feat_blob[basic_pos+0] + grid0) * stride;
        float y_center = (feat_blob[basic_pos+1] + grid1) * stride;
        float w = exp(feat_blob[basic_pos+2]) * stride;
        float h = exp(feat_blob[basic_pos+3]) * stride;
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;

        float box_objectness = feat_blob[basic_pos+4];
        for (int class_idx = 0; class_idx < num_class; class_idx++)
        {
            float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > prob_threshold)
            {
                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = class_idx;
                obj.prob = box_prob;

                objects.push_back(obj);
            }
        } // class loop
    } // point anchor loop
}

float* blobFromImage(cv::Mat& img){
    float* blob = new float[img.total()*3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; int(c) < channels; c++)
        for (size_t  h = 0; int(h) < img_h; h++)
            for (size_t w = 0; int(w) < img_w; w++)
                blob[c * img_w * img_h + h * img_w + w] = (float)img.at<cv::Vec3b>(h, w)[c];
    return blob;
}

static void decode_outputs(float* prob, std::vector<Object>& objects, float scale, const int img_w, const int img_h, float conf_thres, float nms_thres, int img_size, int num_class) {
    std::vector<Object> proposals;
    std::vector<int> strides = {8, 16, 32};
    std::vector<GridAndStride> grid_strides;
    generate_grids_and_stride(strides, grid_strides, img_size);
    generate_yolox_proposals(grid_strides, prob,  conf_thres, proposals, num_class);
    //std::cout << "num of boxes before nms: " << proposals.size() << std::endl;

    qsort_descent_inplace(proposals);

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_thres);


    int count = picked.size();

    //std::cout << "num of boxes: " << count << std::endl;

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x) / scale;
        float y0 = (objects[i].rect.y) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }
}

const float color_list[80][3] =
        {
                {0.000, 0.447, 0.741},
                {0.850, 0.325, 0.098},
                {0.929, 0.694, 0.125},
                {0.494, 0.184, 0.556},
                {0.466, 0.674, 0.188},
                {0.301, 0.745, 0.933},
                {0.635, 0.078, 0.184},
                {0.300, 0.300, 0.300},
                {0.600, 0.600, 0.600},
                {1.000, 0.000, 0.000},
                {1.000, 0.500, 0.000},
                {0.749, 0.749, 0.000},
                {0.000, 1.000, 0.000},
                {0.000, 0.000, 1.000},
                {0.667, 0.000, 1.000},
                {0.333, 0.333, 0.000},
                {0.333, 0.667, 0.000},
                {0.333, 1.000, 0.000},
                {0.667, 0.333, 0.000},
                {0.667, 0.667, 0.000},
                {0.667, 1.000, 0.000},
                {1.000, 0.333, 0.000},
                {1.000, 0.667, 0.000},
                {1.000, 1.000, 0.000},
                {0.000, 0.333, 0.500},
                {0.000, 0.667, 0.500},
                {0.000, 1.000, 0.500},
                {0.333, 0.000, 0.500},
                {0.333, 0.333, 0.500},
                {0.333, 0.667, 0.500},
                {0.333, 1.000, 0.500},
                {0.667, 0.000, 0.500},
                {0.667, 0.333, 0.500},
                {0.667, 0.667, 0.500},
                {0.667, 1.000, 0.500},
                {1.000, 0.000, 0.500},
                {1.000, 0.333, 0.500},
                {1.000, 0.667, 0.500},
                {1.000, 1.000, 0.500},
                {0.000, 0.333, 1.000},
                {0.000, 0.667, 1.000},
                {0.000, 1.000, 1.000},
                {0.333, 0.000, 1.000},
                {0.333, 0.333, 1.000},
                {0.333, 0.667, 1.000},
                {0.333, 1.000, 1.000},
                {0.667, 0.000, 1.000},
                {0.667, 0.333, 1.000},
                {0.667, 0.667, 1.000},
                {0.667, 1.000, 1.000},
                {1.000, 0.000, 1.000},
                {1.000, 0.333, 1.000},
                {1.000, 0.667, 1.000},
                {0.333, 0.000, 0.000},
                {0.500, 0.000, 0.000},
                {0.667, 0.000, 0.000},
                {0.833, 0.000, 0.000},
                {1.000, 0.000, 0.000},
                {0.000, 0.167, 0.000},
                {0.000, 0.333, 0.000},
                {0.000, 0.500, 0.000},
                {0.000, 0.667, 0.000},
                {0.000, 0.833, 0.000},
                {0.000, 1.000, 0.000},
                {0.000, 0.000, 0.167},
                {0.000, 0.000, 0.333},
                {0.000, 0.000, 0.500},
                {0.000, 0.000, 0.667},
                {0.000, 0.000, 0.833},
                {0.000, 0.000, 1.000},
                {0.000, 0.000, 0.000},
                {0.143, 0.143, 0.143},
                {0.286, 0.286, 0.286},
                {0.429, 0.429, 0.429},
                {0.571, 0.571, 0.571},
                {0.714, 0.714, 0.714},
                {0.857, 0.857, 0.857},
                {0.000, 0.447, 0.741},
                {0.314, 0.717, 0.741},
                {0.50, 0.5, 0}
        };

int get_class_num(string file_name) {
    ifstream in(file_name);
    int num=0;
    string line;
    if (in) while (getline(in, line)) num++;
    return num;
}

void doInference(IExecutionContext& context, float* input, float* output, const int output_size, cv::Size input_shape) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);

    assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
    //int mBatchSize = engine.getMaxBatchSize();

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], 3 * input_shape.height * input_shape.width * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], output_size*sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(1, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

stringstream getCommand(string ip, int port, int w, int h, int fps, string name, string b) {

    char url[100];
    sprintf(url, "rtmp://%s:%d/%s", ip.c_str(), port, name.c_str());
    string rtmp_server_url = url;


    char imgSize[20];
    sprintf(imgSize, "-s %dx%d ", w, h);
    string imgsize = imgSize;


    char FPS[20];
    sprintf(FPS, "-r %d ", fps);
    string Fps = FPS;

    char bitrate0[100];
    sprintf(bitrate0, "-b %s ", b.c_str());
    string BitRate = bitrate0;
//    cout << BitRate <<endl;

    stringstream command;
    command << "ffmpeg ";

    // infile options
    command << "-y "  // overwrite output files
            << "-an " // disable audio
            //<< "-c copy "
            << "-f rawvideo " // force format to rawvideo
            << "-vcodec rawvideo "  // force video rawvideo ('copy' to copy stream)
            << "-pix_fmt bgr24 "  // set pixel format to bgr24
            << imgsize  // set frame size (WxH or abbreviation)
            << Fps; // set frame rate (Hz value, fraction or abbreviation)

    command << "-i - "; //

    // outfile options
    command << "-c:v libx264 "  // Hyper fast Audio and Video encoder
            << "-pix_fmt yuv420p "  // set pixel format to yuv420p
            << "-preset ultrafast " // set the libx264 encoding preset to ultrafast
            << "-f flv " // force format to flv
            << "-g 5 "
            << BitRate
            << rtmp_server_url;

    return command;

}

int post_request(string ip, int port, string request_name, string msg) {
    // init
    CURL* curl_conn_ = curl_easy_init();
    if (!curl_conn_) {
        cout<<"init failed"<<endl;
        return -1;
    }

    //
    char url[200];
    sprintf(url, "http://%s:%d/%s", ip.c_str(), port, request_name.c_str());
    curl_easy_setopt(curl_conn_, CURLOPT_CUSTOMREQUEST, "POST");
    curl_easy_setopt(curl_conn_, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl_conn_, CURLOPT_URL, url);
    curl_easy_setopt(curl_conn_, CURLOPT_DEFAULT_PROTOCOL, "http");
    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl_conn_, CURLOPT_HTTPHEADER, headers);

    auto code = curl_easy_setopt(curl_conn_, CURLOPT_POSTFIELDS, msg.c_str());
    if (code != CURLE_OK) {
        cout << "failed to set CURLOPT_POSTFIELDS " << code << endl;
        return -1;
    }

    code = curl_easy_perform(curl_conn_);
    if (code != CURLE_OK) {
        cout << "failed to post " << url <<" code: " << code << endl;
        return -2;
    }
    curl_easy_cleanup(curl_conn_);
    //delete curl_conn_;
    return 0;
}


int main(int argc, char** argv) {

    float conf_thres=BBOX_CONF_THRESH;
    float nms = NMS_THRESH;
    int img_size=640;
    int num_class = NUM_CLASSES;

    std::string input_image_path;
    std::string video_path;
    std::string engine_file_path;
    int sourcetype = -1;
    int i = 1;

    string ip="127.0.0.1";
    string rtmp_name = "live/test";
    string post_name = "detect_result";

    int port = 1935;
    int post_port = 80;
    int fps = 30;

    int source_flag = 0;
    int engine_flag = 0;

    string class_file_name="classes.txt";
    string bitrate="4000000";

    bool repeat = false;
    bool push = true;
    bool show = false;
    bool post = false;

    int wait_time = 1;


    std::string help_string = "\nFormat: -[Options] [Params]\n\n" \
                                "Options     Params\n" \
                                "-e          /path/to/your/tensorrt_engine_file\n" \
                                "-no-push    stop pushing rtmp stream\n" \
                                "-post       post json result(default false)\n" \
                                "-show       show image(default false)\n" \
                                "-repeat     repeat playing video(default false)\n" \
                                /*"-i          /path/to/your/image_file\n" \*/
                                "-v          /path/to/your/video_file or rtsp/rtmp stream\n" \
                                "-conf       confidence threshold between 0-1(default 0.25)\n" \
                                "-nms        NMS threshold between 0-1(default 0.45)\n" \
                                "-size       input size of images, wrong size will cause error(default 640, 416 if tiny-model or nano-model)\n" \
                                "-ip         rtmp server ip(default 127.0.0.1)\n" \
                                "-port       rtmp server port(default 1935)\n" \
                                "-post-port  request server port(default 80)\n" \
                                "-fps        stream rate(default 30)\n" \
                                "-b          bit rate(default 4000000)\n" \
                                "-name       rtmp name(default live/test)\n" \
                                "-post-name  post json result name(default detect_result)\n" \
                                "-clsfile    class file name(default classes.txt)\n" \
                                "-h          show this help\n\n";


    while(i<argc) {
        //std::cout << argv[i];
        if (std::string(argv[i]) == "-h") {
            std::cout<< help_string;
            return 0;
        }
        if (std::string(argv[i]) == "-repeat") {repeat = true; i--;}
        else if (std::string(argv[i]) == "-no-push") {push = false; i--;}
        else if (std::string(argv[i]) == "-show") {show = true; i--;}
        else if (std::string(argv[i]) == "-post") {post = true; i--;}
        else if (i == argc-1){
            std::cout<<"WRONG INPUT PARAMS!\n\n" << help_string;
            return 0;
        }
        if (std::string(argv[i]) == "-i") {sourcetype=0;const std::string source_path {argv[i+1]};input_image_path=source_path;source_flag++;}
        else if (std::string(argv[i]) == "-v") {sourcetype=1;const std::string source_path {argv[i+1]};video_path=source_path;source_flag++;}
        else if (std::string(argv[i]) == "-e") {const std::string engine_path {argv[i+1]};engine_file_path = engine_path;engine_flag++;}
        else if (std::string(argv[i]) == "-conf") {const char* ct {argv[i+1]};conf_thres = atof(ct);}
        else if (std::string(argv[i]) == "-nms") {const char* nt {argv[i+1]};nms = atof(nt);}
        else if (std::string(argv[i]) == "-size") {const char* nt {argv[i+1]};img_size = atoi(nt);}
        else if (std::string(argv[i]) == "-ip") {const std::string ipt {argv[i+1]};ip = ipt;}
        else if (std::string(argv[i]) == "-b") {const std::string bt {argv[i+1]};bitrate = bt;}
        else if (std::string(argv[i]) == "-port") {const char* nt {argv[i+1]};port = atoi(nt);}
        else if (std::string(argv[i]) == "-name") {const std::string rnt {argv[i+1]};rtmp_name = rnt;}
        else if (std::string(argv[i]) == "-post-name") {const std::string rnt {argv[i+1]};post_name = rnt;}
        else if (std::string(argv[i]) == "-post-port") {const char* nt {argv[i+1]};post_port = atoi(nt);}
        else if (std::string(argv[i]) == "-clsfile") {const std::string clsft {argv[i+1]};class_file_name = clsft;}
        else if (std::string(argv[i]) == "-fps") {const char* fpst {argv[i+1]};fps = atoi(fpst);}
        i+=2;
    }

//    cout<<bitrate<<endl;

    bool flag=true;
    if (engine_flag==0) {
        std::cout<<"You should provide Engine File Path!\n";
        flag = false;
    }
    if (source_flag==0) {
        std::cout<<"You should provide Image Source!\n";
        flag = false;
    }
    if (source_flag>1) {
        std::cout<<"Too many Image Sources! You should only input one Image Source!\n";
        flag = false;
    }
    if (engine_flag>1) {
        std::cout<<"Too many Engine File Paths! You should only input one Engine File Path!\n";
        flag = false;
    }
    if (!flag) {
        std::cout<< "\n" <<help_string;
        return 0;
    }
    //std::cout << video_path;

    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    ifstream file(engine_file_path, ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }


    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    auto out_dims = engine->getBindingDimensions(1);
    auto output_size = 1;
    for(int j=0;j<out_dims.nbDims;j++) {
        output_size *= out_dims.d[j];
    }
    static float* prob = new float[output_size];


    cv::Mat img;


    stringstream command;

    bool isfirst = true;


    num_class = get_class_num(class_file_name);


    //cout << num_class <<endl;
    if (num_class == 0) return -1;
    string class_names[num_class+1];

    ifstream in(class_file_name);
    //string line;
    int num=0;
    while (!in.eof())
        getline(in, class_names[num++], '\n');


    if(sourcetype==0){
        img = cv::imread(input_image_path);
        int img_w = img.cols;
        int img_h = img.rows;
        cv::Mat pr_img = static_resize(img, img_size);

        float* blob = blobFromImage(pr_img);
        float scale = std::min(img_size / (img.cols*1.0), img_size / (img.rows*1.0));

        //auto start = std::chrono::system_clock::now();
        doInference(*context, blob, prob, output_size, pr_img.size());
        //auto end = std::chrono::system_clock::now();
        //std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        std::vector<Object> objects;
        decode_outputs(prob, objects, scale, img_w, img_h, conf_thres, nms, img_size, num_class);
        //update_objects(pr_img, objects, fp);
        //cv::waitKey(0);
    }
    else if (sourcetype == 1){
        //std::cout<<"start video\n";

        int img_w;
        int img_h;
        //int push_size = 0;
        //int push_count = 0;
        int baseLine = 0;
        int x = 0;
        int y = 0;
        int out_point_y = 0;
        int key = 0;

        float scale;
        float c_mean;

        auto start = std::chrono::system_clock::now();

        cv::VideoCapture cam(video_path);
        cv::Mat pr_img;
        cv::Mat image;
        cv::Scalar color;
        cv::Scalar txt_color;
        cv::Scalar txt_bk_color;
        cv::Size label_size;


        FILE *fp = nullptr;

        while (cam.isOpened()){

            cam >> img;
            //cv::Mat ori = img.clone();
            //cv::resize(ori, img, cv::Size(1920, 1080));


            img_w = img.cols;
            img_h = img.rows;

            if (img_w * img_h < 1){
                cam.release();
                if (repeat) {
                    cam.open(video_path);
                    continue;
                }
                break;
            }
            pr_img = static_resize(img, img_size);

            float* blob = blobFromImage(pr_img);
            scale = std::min(img_size / (img.cols*1.0), img_size / (img.rows*1.0));

            doInference(*context, blob, prob, output_size, pr_img.size());


            std::vector<Object> objects;
            decode_outputs(prob, objects, scale, img_w, img_h, conf_thres, nms, img_size, num_class);

            if (isfirst && push) {
                //push_size = img_w * img_h * 3;
                command = getCommand(ip, port, img_w, img_h, fps, rtmp_name, bitrate);
                isfirst = false;
                cout<<"open pipe..."<<endl;
                fp = popen(command.str().c_str(), "w");
                cout<<"pipe opened!!"<<endl;
            }

            while (float(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count()) * fps < 965)
                continue;
            start = std::chrono::system_clock::now();

            image = img.clone();
//            cv::Mat out;
            if (objects.size() > 0) {

                char result_text[objects.size() * 150];

                for (size_t i = 0; i < objects.size(); i++)
                {
                    const Object& obj = objects[i];

                    if (post) {
                        if (i>0) {
                            sprintf(
                                    result_text,
                                    "%s, {\"loc\": [%d, %d, %d, %d], \"label\": \"%s\", \"conf\": %.3f}",
                                    result_text,
                                    int(obj.rect.x),
                                    int(obj.rect.y),
                                    int(obj.rect.width),
                                    int(obj.rect.height),
                                    class_names[obj.label].c_str(),
                                    obj.prob
                            );
                        } else {
                            sprintf(
                                    result_text,
                                    "[{\"loc\": [%d, %d, %d, %d], \"label\": \"%s\",\"conf\": %.3f}",
                                    int(obj.rect.x),
                                    int(obj.rect.y),
                                    int(obj.rect.width),
                                    int(obj.rect.height),
                                    class_names[obj.label].c_str(),
                                    obj.prob
                            );
                        }

                    }




                    color = cv::Scalar(color_list[obj.label][0], color_list[obj.label][1], color_list[obj.label][2]);
                    c_mean = cv::mean(color)[0];

                    if (c_mean > 0.5){
                        txt_color = cv::Scalar(0, 0, 0);
                    } else {
                        txt_color = cv::Scalar(255, 255, 255);
                    }
                    cv::rectangle(image, obj.rect, color * 255, 2);

                    char text[256];
                    sprintf(text, "%s %.1f%%", class_names[obj.label].c_str(), obj.prob * 100);

                    baseLine = 0;
                    label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

                    txt_bk_color = color * 0.7 * 255;

                    x = obj.rect.x;
                    y = obj.rect.y + 1;
                    //int y = obj.rect.y - label_size.height - baseLine;
                    if (y > image.rows)
                        y = image.rows;
                    //if (x + label_size.width > image.cols)
                    //x = image.cols - label_size.width;

                    out_point_y = y - label_size.height - baseLine;
                    if (out_point_y >= 0) y = out_point_y;

                    cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                                  txt_bk_color, -1);

//                    cv::addWeighted(img, 0.5, image, 0.5, 1, out);

//                    cv::putText(img, text, cv::Point(x, y + label_size.height),
//                                cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
                    cv::putText(image, text, cv::Point(x, y + label_size.height),
                                cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
                }


                if (post) {
                    char out[strlen(result_text)+100];
                    sprintf(out, "{\"%s\": %s]}", "outputs", result_text);
                    string ret = out;
                    post_request(ip, post_port, post_name, ret);
                }
                //delete result_text;
            }
            else if (post) {
                string ret = "{\"outputs\": []}";
                post_request(ip, post_port, post_name, ret);
            }



            if (show) {
                cv::imshow("image", image);
                key = cv::waitKey(wait_time);
                if (key == 27) {
                    cv::destroyAllWindows();
                    break;
                } else if (key == 32 && !push) {
                    wait_time = 1 - wait_time;
                }

            }
            if (push)
                fwrite(image.data, sizeof(char), image.total() * image.elemSize(), fp);

            delete blob;

        }
        if (push) pclose(fp);
    }
    else return -1;

    // delete the pointer to the float
    // delete blob;

    // destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    //cv::destroyAllWindows();
    return 0;
}
