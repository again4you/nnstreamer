
#include <glib.h>
#include <string.h>
#include <tensor_filter_custom.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_log.h>

#include <iostream>
#include <memory>
#include <vector>

#include <NvInfer.h>
#include <NvUffParser.h>
#include <NvInferRuntimeCommon.h>
#include <cuda_runtime_api.h>

using Severity = nvinfer1::ILogger::Severity;

/* @brief a global object of ILogger */
class Logger : public nvinfer1::ILogger
{
	void log(Severity severity, const char* msg) override
	{
    switch (severity) {
      case Severity::kWARNING:
        ml_logw ("%s", msg);
        break;
      case Severity::kINFO:
        ml_logi ("%s", msg);
        break;
      case Severity::kVERBOSE:
        ml_logd ("%s", msg);
        break;
      default:
        ml_loge ("%s", msg);
        break;
    }
	}
} gLogger;

struct InferDeleter
{
  template <typename T>
  void operator()(T* obj) const
  {
    if (obj)
      obj->destroy();
  }
};

class TensorRTCore
{
  template <typename T>
  using UniquePtr = std::unique_ptr<T, InferDeleter>;

public:
  TensorRTCore (const char * _uff_path);
  ~TensorRTCore ();

  int initTensorInfo(const GstTensorFilterProperties * prop);

  int buildEngine();
  int infer(const GstTensorMemory * inputData, GstTensorMemory * outputData);

  int getInputTensorDim(GstTensorsInfo * info);
  int getOutputTensorDim(GstTensorsInfo * info);

private:
  char * _uff_path;

  GstTensorsInfo _inputTensorMeta;
  GstTensorsInfo _outputTensorMeta;

  template <typename T>
  UniquePtr<T> makeUnique(T* t)
  {
    return UniquePtr<T>{t};
  }
};

void init_filter_tensorrt (void) __attribute__ ((constructor));
void fini_filter_tensorrt (void) __attribute__ ((destructor));

TensorRTCore::TensorRTCore (const char * uff_path)
{
  _uff_path = g_strdup (uff_path);

  gst_tensors_info_init (&_inputTensorMeta);
  gst_tensors_info_init (&_outputTensorMeta);
}

TensorRTCore::~TensorRTCore ()
{
  gst_tensors_info_free (&_inputTensorMeta);
  gst_tensors_info_free (&_outputTensorMeta);

  g_free (_uff_path);
}

int
TensorRTCore::initTensorInfo(const GstTensorFilterProperties * prop)
{
  /* TODO param check */
  
  gst_tensors_info_copy (&_inputTensorMeta, &prop->input_meta);
  gst_tensors_info_copy (&_outputTensorMeta, &prop->output_meta);

  return 0;
}

int
TensorRTCore::buildEngine()
{
  auto builder = makeUnique(nvinfer1::createInferBuilder(gLogger));
  if (!builder) {
    ml_loge ("Failed to create builder");
    return -1;
  }

  auto network = makeUnique(builder->createNetworkV2(0U));
  if (!network) {
    ml_loge ("Failed to create network");
    return -1;
  }

  auto config = makeUnique(builder->createBuilderConfig());
  if (!config) {
    ml_loge ("Failed to create config");
    return -1;
  }

  auto parser = makeUnique(nvuffparser::createUffParser());
  if (!parser) {
    ml_loge ("Failed to create parser");
    return -1;
  }

  /* Register tensor input & output */
  parser->registerInput("input",
    
    nvuffparser::UffInputOrder::kNCHW);
  
  
  return 0;
}

int
TensorRTCore::infer(const GstTensorMemory * inputData,
  GstTensorMemory * outputData)
{
  return 0;
}

int
TensorRTCore::getInputTensorDim(GstTensorsInfo * info)
{
  gst_tensors_info_copy (info, &_inputTensorMeta);
  return 0;
}

int
TensorRTCore::getOutputTensorDim(GstTensorsInfo * info)
{
  gst_tensors_info_copy (info, &_outputTensorMeta);
  return 0;
}



/************************************************************/
static int
tensorrt_loadModel (const GstTensorFilterProperties * prop, void **private_data)
{
  TensorRTCore * core;
  const gchar * uff_file;

  if (prop->num_models != 1) {
    ml_loge ("TensorRT filter requires one UFF model file\n");
    return -1;
  }

  /* TODO check existing core*/

  uff_file = prop->model_files[0];
  if (uff_file == nullptr) {
    ml_loge ("UFF model file is not valid\n");
    return -1;
  }


  /* TODO use try/catch block */
  core = new TensorRTCore(uff_file);
  if (core == nullptr) {
    ml_loge ("Failed to allocate memory for filter subplugin: TensorRT\n");
    return -1;
  }

  /* TODO init internal info */
  if (core->initTensorInfo(prop) != 0) {
    *private_data = nullptr;
    delete core;

    ml_loge ("Failed to initialize an object: TensorRT\n");
    return -2;
  }

  /* TODO make TensorRT object */
  core->buildEngine();

  *private_data = core;

  return 0;
}




/************************************************************/

static int
tensorrt_open (const GstTensorFilterProperties * prop, void **private_data)
{
  int status = tensorrt_loadModel (prop, private_data);

  return status;
}

static void
tensorrt_close (const GstTensorFilterProperties * prop, void **private_data)
{
  TensorRTCore *core = static_cast<TensorRTCore *> (*private_data);
  if (!core)
    return;

  delete core;
  *private_data = nullptr;
}

static int
tensorrt_infer (const GstTensorFilterProperties * prop, void **private_data,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  TensorRTCore *core = static_cast<TensorRTCore *> (*private_data);

  /* TODO need to make */


  return core->infer (input, output);
}

static int
tensorrt_getInputDim (const GstTensorFilterProperties * prop,
  void **private_data, GstTensorsInfo * info)
{
  TensorRTCore *core = static_cast<TensorRTCore *> (*private_data);

  /* TODO check core */

  return core->getInputTensorDim (info);
}

static int
tensorrt_getOutputDim (const GstTensorFilterProperties * prop,
  void **private_data, GstTensorsInfo * info)
{
  TensorRTCore *core = static_cast<TensorRTCore *> (*private_data);

  /* TODO check core */

  return core->getOutputTensorDim (info);
}

static void
tensorrt_destroyNotify (void **private_data, void *data)
{
  /* Do nothing */
}

static int
tensorrt_checkAvailability (accl_hw hw)
{
  return 0;
}

static gchar filter_subplugin_tensorrt[] = "tensorrt";

static GstTensorFilterFramework NNS_support_tensorrt = {
  .version = GST_TENSOR_FILTER_FRAMEWORK_V0,
  .open = tensorrt_open,
  .close = tensorrt_close,
};

void
init_filter_tensorrt (void)
{
  NNS_support_tensorrt.name = filter_subplugin_tensorrt;
  NNS_support_tensorrt.allow_in_place = FALSE;
  NNS_support_tensorrt.allocate_in_invoke = TRUE;
  NNS_support_tensorrt.run_without_model = FALSE;
  NNS_support_tensorrt.verify_model_path = FALSE;
  NNS_support_tensorrt.invoke_NN = tensorrt_infer;
  NNS_support_tensorrt.getInputDimension = tensorrt_getInputDim;
  NNS_support_tensorrt.getOutputDimension = tensorrt_getOutputDim;
  NNS_support_tensorrt.destroyNotify = tensorrt_destroyNotify;
  NNS_support_tensorrt.checkAvailability = tensorrt_checkAvailability;

  nnstreamer_filter_probe (&NNS_support_tensorrt);
}

void
fini_filter_tensorrt (void)
{
  nnstreamer_filter_exit (NNS_support_tensorrt.name);
}
