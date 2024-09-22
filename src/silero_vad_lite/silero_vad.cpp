#include <codecvt>
#include <iostream>
#include <locale>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

// Helper function to convert std::string (in UTF-8) to std::wstring, which is required for Windows
std::wstring string_to_wstring(const std::string& str) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.from_bytes(str);
}

class SileroVAD {
private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::shared_ptr<Ort::Session> session = nullptr;
    Ort::MemoryInfo memory_info;

    std::vector<Ort::Value> ort_inputs;

    std::vector<const char *> ort_input_node_names = {"input", "state", "sr"};
    std::vector<float> ort_state;
    std::vector<int64_t> ort_sample_rate;

    int64_t ort_input_node_shape[2] = {1, 0};  // Element 1 will be set to window_size_samples
    const int64_t ort_state_node_shape[3] = {2, 1, 128};
    const int64_t ort_sample_rate_node_shape[1] = {1};

    std::vector<Ort::Value> ort_outputs;
    std::vector<const char *> output_node_names = {"output", "stateN"};

    void init_engine_threads(int inter_threads, int intra_threads)
    {
        // The method should be called in each thread/proc in multi-thread/proc work
        session_options.SetIntraOpNumThreads(intra_threads);
        session_options.SetInterOpNumThreads(inter_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    };

public:
    const size_t window_size_samples;

    SileroVAD(const std::string& model_path, int sample_rate) :
        env(ORT_LOGGING_LEVEL_WARNING, "SileroVAD"),
        memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU)),
        ort_state(2 * 1 * 128),
        ort_sample_rate(1, sample_rate),
        window_size_samples(32 * (sample_rate / 1000))  // NOTE: ~32 ms * sample_rate_per_ms: 512 samples for 16 kHz, 256 samples for 8 kHz
    {
        if (sample_rate != 16000 && sample_rate != 8000) {
            throw std::invalid_argument("Sample rate must be 16000 or 8000");
        }

        init_engine_threads(1, 1);
        #ifdef _WIN32
            const ORTCHAR_T* model_path_ort = string_to_wstring(model_path).c_str();
        #else
            const ORTCHAR_T* model_path_ort = model_path.c_str();
        #endif
        session = std::make_shared<Ort::Session>(env, model_path_ort, session_options);
        std::cerr << "Model loaded" << std::endl;
        std::cerr << "Model loaded: " << session.get() << std::endl;
        ort_input_node_shape[1] = window_size_samples;
    }

    // Run model to compute speech probability of exactly one window
    float predict(float* data, size_t size) {
        if (size != window_size_samples) {
            throw std::invalid_argument("Input size must be equal to window_size_samples");
        }
        Ort::Value input_ort = Ort::Value::CreateTensor<float>(memory_info, data, size, ort_input_node_shape, 2);
        Ort::Value state_ort = Ort::Value::CreateTensor<float>(memory_info, ort_state.data(), ort_state.size(), ort_state_node_shape, 3);
        Ort::Value sr_ort = Ort::Value::CreateTensor<int64_t>(memory_info, ort_sample_rate.data(), ort_sample_rate.size(), ort_sample_rate_node_shape, 1);

        ort_inputs.clear();
        ort_inputs.emplace_back(std::move(input_ort));
        ort_inputs.emplace_back(std::move(state_ort));
        ort_inputs.emplace_back(std::move(sr_ort));

        ort_outputs = session->Run(
            Ort::RunOptions{nullptr},
            ort_input_node_names.data(), ort_inputs.data(), ort_inputs.size(),
            output_node_names.data(), output_node_names.size());

        float speech_prob = ort_outputs[0].GetTensorMutableData<float>()[0];
        float *stateN_output = ort_outputs[1].GetTensorMutableData<float>();
        std::memcpy(ort_state.data(), stateN_output, ort_state.size() * sizeof(float));

        return speech_prob;
    }
};

#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API
#endif

extern "C" {
    EXPORT_API SileroVAD* SileroVAD_new(const char* model_path, int sample_rate) {
        return new SileroVAD(model_path, sample_rate);
    }

    EXPORT_API void SileroVAD_delete(SileroVAD* vad) {
        delete vad;
    }

    EXPORT_API float SileroVAD_process(SileroVAD* vad, float* data, size_t size) {
        return vad->predict(data, size);
    }

    EXPORT_API size_t SileroVAD_get_window_size_samples(SileroVAD* vad) {
        return vad->window_size_samples;
    }
}
