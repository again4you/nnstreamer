/**
 * NNStreamer TensorRT Custom Filter Example: Reshape
 *
 * Copyright (C) 2020 Dongju Chae <dongju.chae@samsung.com>
 *
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * @file  nnstreamer_customfilter_example_tensorrt_reshape.cc
 * @date  14 Sep 2020
 * @brief  TensorRT Custom NNStreamer Filter Example: Reshape
 * @author  Dongju Chae <dongju.chae@samsung.com>
 * @bug  No known bugs
 *
 * Reference code: sampleDynamicReshape.cpp in TensorRT samples
 *
 * This example reshapes the inputs to the given dimensions.
 * The custom property is to be given as, "custom=D1:D2:D3:D4"
 * E.g., custom=224:224:3:1
 *
 */

#include <glib.h>
#include <string.h>
#include <tensor_filter_custom.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_log.h>

#include <iostream>
#include <memory>
#include <vector>

#include <NvInfer.h>
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

class CustomTensorRT
{
  template <typename T>
  using UniquePtr = std::unique_ptr<T, InferDeleter>;

  public:
    CustomTensorRT();
    ~CustomTensorRT();

    bool buildEngine();

    void setInputMeta(const GstTensorsInfo *info);
    void setOutputMeta(const GstTensorsInfo *info);

    const GstTensorsInfo * getInputMeta() { return &mInputMeta; }
    const GstTensorsInfo * getOutputMeta() { return &mOutputMeta; }

    bool resize();
    bool infer(const GstTensorMemory * input, GstTensorMemory * output);

  private:
    GstTensorsInfo mInputMeta;
    GstTensorsInfo mOutputMeta;

    nvinfer1::Dims mInputDims;
    nvinfer1::Dims mOutputDims;

    void * mInput;
    void * mOutput;

    UniquePtr<nvinfer1::ICudaEngine> mEngine{nullptr};
    UniquePtr<nvinfer1::IExecutionContext> mContext{nullptr};

    template <typename T>
    UniquePtr<T> makeUnique(T* t)
    {
      return UniquePtr<T>{t};
    }
};

CustomTensorRT::CustomTensorRT ()
{
  gst_tensors_info_init (&mInputMeta);
  gst_tensors_info_init (&mOutputMeta);

  mInput = nullptr;
  mOutput = nullptr;
}

CustomTensorRT::~CustomTensorRT ()
{
  gst_tensors_info_free (&mInputMeta);
  gst_tensors_info_free (&mOutputMeta);

  if (mInput != nullptr)
    cudaFree (&mInput);

  if (mOutput != nullptr)
    cudaFree (&mOutput);
}

void
CustomTensorRT::setInputMeta (const GstTensorsInfo *info)
{
  gst_tensors_info_copy (&mInputMeta, info);

  /* TensorRT uses the NCHW data format */
  mInputDims = nvinfer1::Dims4 {
    (int) mInputMeta.info[0].dimension[3],
    (int) mInputMeta.info[0].dimension[2],
    (int) mInputMeta.info[0].dimension[1],
    (int) mInputMeta.info[0].dimension[0]
  };
}

void
CustomTensorRT::setOutputMeta (const GstTensorsInfo *info)
{
  gst_tensors_info_copy (&mOutputMeta, info);

  /* TensorRT uses the NCHW data format */
  mOutputDims = nvinfer1::Dims4 {
    (int) mOutputMeta.info[0].dimension[3],
    (int) mOutputMeta.info[0].dimension[2],
    (int) mOutputMeta.info[0].dimension[1],
    (int) mOutputMeta.info[0].dimension[0]
  };
}

bool
CustomTensorRT::buildEngine()
{
  auto builder = makeUnique(nvinfer1::createInferBuilder(gLogger));
  if (!builder) {
    ml_loge ("Failed to create builder");
    return false;
  }

  auto network = makeUnique(
      builder->createNetworkV2(
        1U << static_cast<uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
  if (!network) {
    ml_loge ("Failed to create model");
    return false;
  }

  /* dynamic input shape */
  auto input = network->addInput("input", nvinfer1::DataType::kFLOAT,
      nvinfer1::Dims4{1, 3, -1, -1});
  auto resize = network->addResize(*input);

  resize->setOutputDimensions(mOutputDims);
  network->markOutput(*resize->getOutput(0));

  auto config = makeUnique(builder->createBuilderConfig());
  if (!config) {
    ml_loge ("Failed to create builder config");
    return false;
  }

  /* specifies a range of input dimensions */
  auto profile = builder->createOptimizationProfile();
  profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
      nvinfer1::Dims4{1,3,1,1});
  profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
      nvinfer1::Dims4{1,3,480,640});
  profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
      nvinfer1::Dims4{1,3,960,1280});
  config->addOptimizationProfile(profile);

  config->setMaxWorkspaceSize(16 * (1 << 20));
  builder->setMaxBatchSize(1);

  mEngine = makeUnique(builder->buildEngineWithConfig(*network, *config));
  if (!mEngine) {
    ml_loge ("Failed to create builder config");
    return false;
  }

  mContext = makeUnique(mEngine->createExecutionContext());
  if (!mContext) {
    ml_loge ("Failed to create execution context");
    return false;
  }

  return true;
}

bool
CustomTensorRT::resize()
{
  gsize input_size = gst_tensor_info_get_size (&mInputMeta.info[0]);
  gsize output_size = gst_tensor_info_get_size (&mOutputMeta.info[0]);

  if (mInput != nullptr) {
    cudaFree (&mInput);
    mInput = nullptr;
  }

  if (mOutput != nullptr) {
    cudaFree (&mOutput);
    mOutput = nullptr;
  }

  if (cudaMalloc (&mInput, input_size) != cudaSuccess) {
    ml_loge ("Failed to allocate GPU memory");
    return false;
  }

  if (cudaMalloc (&mOutput, output_size) != cudaSuccess) {
    ml_loge ("Failed to allocate GPU memory");
    return false;
  }

  return true;
}

bool
CustomTensorRT::infer(const GstTensorMemory * input, GstTensorMemory * output)
{
  if (cudaMemcpy(mInput, input->data, input->size,
        cudaMemcpyHostToDevice) != cudaSuccess) {
    ml_loge ("Failed to feed input data to GPU");
    return false;
  }

  /* set the input dimensions */
  if (!mContext->setBindingDimensions(0, mInputDims)) {
    ml_loge ("Failed to set binding dimensions");
    return false;
  }

  /* all dynamic input dimensions are specified */
  if (!mContext->allInputDimensionsSpecified()) {
    ml_loge ("Not all input dimensions are specified");
    return false;
  }

  std::vector<void*> bindings = {mInput, mOutput};
  if (!mContext->executeV2(bindings.data())) {
    ml_loge ("Failed to execute the network");
    return false;
  }

  if (cudaMemcpy(output->data, mOutput, output->size,
        cudaMemcpyDeviceToHost) != cudaSuccess) {
    ml_loge ("Failed to retrieve output data from GPU");
    return false;
  }

  return true;
}

/**
 * @brief init callback of tensor_filter custom
 */
static void *
pt_init (const GstTensorFilterProperties * prop)
{
  CustomTensorRT * trt = new CustomTensorRT;
  GstTensorsInfo info;

  gst_tensors_info_init (&info);

  if (prop->custom_properties && strlen (prop->custom_properties) > 0) {
    gchar **strv = g_strsplit (prop->custom_properties, ":", -1);
    gsize i;

    if (g_strv_length (strv) != NNS_TENSOR_RANK_LIMIT) {
      ml_loge ("Please specify a proper 'custom' property");
      goto err;
    }

    info.num_tensors = 1;
    for (i = 0; i < g_strv_length (strv); i++) {
      info.info[0].type = _NNS_FLOAT32;
      info.info[0].dimension[i] = (int) g_ascii_strtoll (strv[i], NULL, 10);
    }

    g_strfreev (strv);
  } else {
    ml_loge ("Please specify 'custom' property");
    goto err;
  }

  trt->setOutputMeta (&info);
  if (!trt->buildEngine ()) {
    ml_loge ("Failed to build a TensorRT engine");
    goto err;
  }

  return trt;

err:
  delete trt;
  return nullptr;
}

/**
 * @brief exit callback of tensor_filter custom
 */
static void
pt_exit (void *private_data, const GstTensorFilterProperties * prop)
{
  CustomTensorRT *trt = static_cast<CustomTensorRT *> (private_data);
  g_assert (trt);

  delete trt;
}

/**
 * @brief setInputDimension callback of tensor_filter custom
 */
static int
set_inputDim (void *private_data, const GstTensorFilterProperties * prop,
    const GstTensorsInfo * in_info, GstTensorsInfo * out_info)
{
  CustomTensorRT *trt = static_cast<CustomTensorRT *> (private_data);
  g_assert (trt);

  trt->setInputMeta (in_info);

  gst_tensors_info_copy (out_info, trt->getOutputMeta ());

  return 0;
}

/**
 * @brief invoke callback of tensor_filter custom
 */
static int
pt_invoke (void *private_data, const GstTensorFilterProperties * prop,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  CustomTensorRT *trt = static_cast<CustomTensorRT *> (private_data);

  if (!trt->resize ())
    return -1;

  if (!trt->infer (input, output))
    return -1;

  return 0;
}

/**
 * @brief tensor_filter custom subplugin definition
 */
static NNStreamer_custom_class NNStreamer_custom_body = {
  .initfunc = pt_init,
  .exitfunc = pt_exit,
  .getInputDim = NULL,
  .getOutputDim = NULL,
  .setInputDim = set_inputDim,
  .invoke = pt_invoke,
  .allocate_invoke = NULL,
  .destroy_notify = NULL,
};

/* The dyn-loaded object */
NNStreamer_custom_class *NNStreamer_custom = &NNStreamer_custom_body;
