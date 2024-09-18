#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <iostream>

class SileroVAD {
private:
    Ort::Env env;
    Ort::Session session;
    Ort::MemoryInfo memory_info;

public:
    SileroVAD(const std::string& model_path) 
        : env(ORT_LOGGING_LEVEL_WARNING, "SileroVAD"),
          session(env, model_path.c_str(), Ort::SessionOptions{nullptr}),
          memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}

    float process(const float* data, int size, int sample_rate) {
        std::vector<float> input_tensor_values(data, data + size);
        std::vector<int64_t> input_shape = {1, size};

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(),
            input_shape.data(), input_shape.size());

        std::vector<Ort::Value> ort_inputs;
        ort_inputs.push_back(std::move(input_tensor));

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr}, 
            {"input"}, 
            ort_inputs.data(), 
            ort_inputs.size(), 
            {"output"});

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        return output_data[0];
    }
};

extern "C" {
    SileroVAD* SileroVAD_new(const char* model_path) {
        return new SileroVAD(model_path);
    }

    void SileroVAD_delete(SileroVAD* vad) {
        delete vad;
    }

    float SileroVAD_process(SileroVAD* vad, float* data, int size, int sample_rate) {
        return vad->process(data, size, sample_rate);
    }
}
