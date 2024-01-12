#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author Yelin Jeong <yelini.jeong@samsung.com>
## @date Jan 08 2024
## @brief SSAT Test Cases for transform padding
##

if [[ "$SSATAPILOADED" != "1" ]]; then
    SILENT=0
    INDEPENDENT=1
    search="ssat-api.sh"
    source $search
    printf "${Blue}Independent Mode${NC}"
fi

# This is compatible with SSAT (https://github.com/myungjoo/SSAT)
testInit $1

PATH_TO_PLUGIN="../../build"

if [ "$SKIPGEN" == "YES" ]; then
    echo "Test Case Generation Skipped"
    sopath=$2
else
    echo "Test Case Generation Started"
    python3 generateTest.py || (echo "Failed to run test preparation script (generateTest.py). Test not available." && report && exit)
    sopath=$1
fi

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=\"test_00.dat\" blocksize=-1 ! application/octet-stream ! tensor_converter input-dim=100:50:3:2:4 input-type=int8 ! tensor_transform mode=padding option=left:1,right:1,layout:NHWC ! filesink location=\"./result_00.dat\" sync=true" 1 0 0 $PERFORMANCE
callCompareTest result_00.dat test_00.dat.golden 1 "Golden test comparison 1" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=\"test_01.dat\" blocksize=-1 ! application/octet-stream ! tensor_converter input-dim=100:50:3:2 input-type=int8 ! tensor_transform mode=padding option=left:1,right:1,top:2,bottom:2 ! filesink location=\"./result_01.dat\" sync=true" 2 0 0 $PERFORMANCE
callCompareTest result_01.dat test_01.dat.golden 2 "Golden test comparison 2" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=\"test_02.dat\" blocksize=-1 ! application/octet-stream ! tensor_converter input-dim=100:50:3 input-type=float32 ! tensor_transform mode=padding option=left:1,right:1,top:1,bottom:1,front:1,back:1 ! filesink location=\"./result_02.dat\" sync=true" 3 0 0 $PERFORMANCE
callCompareTest result_02.dat test_02.dat.golden 3 "Golden test comparison 3" 1 0

# # # Test tensors stream
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
        filesrc location=\"test_02.dat\" blocksize=-1 ! application/octet-stream ! tensor_converter input-dim=100:50:3 input-type=float32 ! tee name=t \
        t. ! queue ! mux.sink_0 \
        t. ! queue ! mux.sink_1 \
        t. ! queue ! mux.sink_2 \
        tensor_mux name=mux ! tensor_transform mode=padding option=left:1,right:1,top:1,bottom:1,front:1,back:1 ! tensor_demux name=demux \
        demux.src_0 ! queue ! filesink location=\"./result_03_0.dat\" sync=true \
        demux.src_1 ! queue ! filesink location=\"./result_03_1.dat\" sync=true \
        demux.src_2 ! queue ! filesink location=\"./result_03_2.dat\" sync=true" 4 0 0 $PERFORMANCE
callCompareTest result_03_0.dat test_02.dat.golden 4 "Golden test comparison 4-0" 1 0
callCompareTest result_03_1.dat test_02.dat.golden 4 "Golden test comparison 4-1" 1 0
callCompareTest result_03_2.dat test_02.dat.golden 4 "Golden test comparison 4-2" 1 0

rm *.golden *.dat

report
