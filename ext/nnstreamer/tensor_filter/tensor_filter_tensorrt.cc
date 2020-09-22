
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

  int getTensorType(tensor_type t);

private:
  char * _uff_path;

  GstTensorsInfo _inputTensorMeta;
  GstTensorsInfo _outputTensorMeta;

  nvinfer1::Dims _InputDims;
  nvinfer1::Dims _OutputDims;
  nvinfer1::DataType _DataType;

  UniquePtr<nvinfer1::ICudaEngine> _Engine{nullptr};
  UniquePtr<nvinfer1::IExecutionContext> _Context{nullptr};

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
TensorRTCore::getTensorType(tensor_type t)
{
  switch (t) {
    case _NNS_INT32:
      _DataType = nvinfer1::DataType::kINT32;
      break;
    case _NNS_FLOAT32:
      _DataType = nvinfer1::DataType::kFLOAT;
      break;
    case _NNS_INT8:
      _DataType = nvinfer1::DataType::kINT8;
      break;

    default:
      /**
       * TensorRT supports kFLOAT(32bit), kHALF(16bit), kINT8, kINT32 and kBOOL.
       * However, NNStreamer does not support kHALF and kBOOL.
       */
      return -1;
  }

  return 0;
}


int
TensorRTCore::initTensorInfo(const GstTensorFilterProperties * prop)
{
  /* TODO param check */
  
  gst_tensors_info_copy (&_inputTensorMeta, &prop->input_meta);
  gst_tensors_info_copy (&_outputTensorMeta, &prop->output_meta);

  /* TensorRT internally uses the NCHW format */
  _InputDims = nvinfer1::Dims4 {
    (int) _inputTensorMeta.info[0].dimension[3],
    (int) _inputTensorMeta.info[0].dimension[2],
    (int) _inputTensorMeta.info[0].dimension[1],
    (int) _inputTensorMeta.info[0].dimension[0]
  };

  _OutputDims = nvinfer1::Dims4 {
    (int) _outputTensorMeta.info[0].dimension[3],
    (int) _outputTensorMeta.info[0].dimension[2],
    (int) _outputTensorMeta.info[0].dimension[1],
    (int) _outputTensorMeta.info[0].dimension[0]
  };

  if (getTensorType(_inputTensorMeta.info[0].type) != 0) {
    ml_loge ("TensorRT filter does not support the input data type.");
    return -1;
  }

  return 0;
}

int
TensorRTCore::buildEngine()
{
  /* Make builder, network, config, parser object */
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
    _InputDims, nvuffparser::UffInputOrder::kNCHW);
  parser->registerOutput("output");

  /* Parse the imported model */
  parser->parse (_uff_path, *network, _DataType);

  /* Set config */
  config->setMaxWorkspaceSize(1 << 20);
  config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);

  /* Create Engine object */
  _Engine = makeUnique(builder->buildEngineWithConfig(*network, *config));
  if (!_Engine) {
    ml_loge ("Failed to create the TensorRT Engine object");
    return -1;
  }

  /* Create ExecutionContext obejct */
  _Context = makeUnique(_Engine->createExecutionContext());
  if (!_Context) {
    ml_loge ("Failed to create the TensorRT ExecutionContext object");
    return -1;
  }

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
