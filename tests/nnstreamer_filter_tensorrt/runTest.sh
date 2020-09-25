#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author Sangjung Woo <sangjung.woo@samsung.com>
## @date Sep 24 2020
## @brief SSAT Test Cases for NNStreamer
##
if [[ "$SSATAPILOADED" != "1" ]]; then
    SILENT=0
    INDEPENDENT=1
    search="ssat-api.sh"
    source $search
    printf "${Blue}Independent Mode${NC}
"
fi

# This is compatible with SSAT (https://github.com/myungjoo/SSAT)
testInit $1

# NNStreamer and plugins path for test
PATH_TO_PLUGIN="../../build"

if [[ -d $PATH_TO_PLUGIN ]]; then
    ini_path="${PATH_TO_PLUGIN}/ext/nnstreamer/tensor_filter"
    if [[ -d ${ini_path} ]]; then
        check=$(ls ${ini_path} | grep tensorrt.so)
        if [[ ! $check ]]; then
            echo "Cannot find TensorRT shared lib"
            report
            exit
        fi
    else
        echo "Cannot find ${ini_path}"
    fi
else
    ini_file="/etc/nnstreamer.ini"
    if [[ -f ${ini_file} ]]; then
        path=$(grep "^filters" ${ini_file})
        key=${path%=*}
        value=${path##*=}

        if [[ $key != "filters" ]]; then
            echo "String Error"
            report
            exit
        fi

        if [[ -d ${value} ]]; then
            check=$(ls ${value} | grep tensorflow2-lite.so)
            if [[ ! $check ]]; then
                echo "Cannot find tensorflow2-lite shared lib"
                report
                exit
            fi
        else
            echo "Cannot file ${value}"
            report
            exit
        fi
    else
        echo "Cannot identify nnstreamer.ini"
        report
        exit
    fi
fi

PATH_TO_MODEL="../test_models/models/lenet5.uff"
PATH_TO_DATA="../test_models/data/0.pgm"

# Passed
#gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_DATA} ! image/x-portable-graymap,width=28,height=28,framerate=0/1 
#    ! pnmdec ! video/x-raw,format=GRAY8 ! tensor_converter input-type=uint8 
#    ! tensor_filter framework=tensorrt model=${PATH_TO_MODEL} input=28:28:1:1 inputtype=int8 inputname=input output=10:1:1:1 outputtype=int8 outputname=output
#    ! filesink location=sj.out.log " 3 0 0 $PERFORMANCE


#gstTest "-v --gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_DATA} ! image/x-portable-graymap,width=28,height=28,framerate=0/1 
#    ! pnmdec ! video/x-raw,format=GRAY8 ! tensor_converter input-type=uint8
#    ! tensor_transform mode=arithmetic option=typecast:float32,div:255.0
#    ! tensor_filter framework=tensorrt model=${PATH_TO_MODEL} input=28:28:1 inputtype=float32 inputname=in output=10:1:1:1 outputtype=float32 outputname=out
#    ! filesink location=sj.out.log " 3 0 0 $PERFORMANCE


#gstTest "-v --gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_DATA} ! image/x-portable-graymap,width=28,height=28,framerate=0/1 
#    ! pnmdec ! video/x-raw,format=GRAY8 ! tensor_converter input-type=uint8
#    ! tensor_transform mode=transpose option=1:2:0:3 
#    ! tensor_transform mode=arithmetic option=typecast:float32,div:255.0
#    ! tensor_filter framework=tensorrt model=${PATH_TO_MODEL} input=28:28:1 inputtype=float32 inputname=in output=10:1:1:1 outputtype=float32 outputname=out
#    ! filesink location=sj.out.log " 3 0 0 $PERFORMANCE

gstTest "-v --gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_DATA} ! image/x-portable-graymap,width=28,height=28,framerate=0/1 
    ! pnmdec ! video/x-raw,format=GRAY8 ! tensor_converter input-type=uint8
    ! tensor_transform mode=transpose option=1:2:0:3 
    ! tensor_transform mode=arithmetic option=typecast:float32,div:-255.0,add:1 
    ! tensor_filter framework=tensorrt model=${PATH_TO_MODEL} input=28:28:1 inputtype=float32 inputname=in output=10:1:1:1 outputtype=float32 outputname=out
    ! filesink location=sj.out.log " 3 0 0 $PERFORMANCE

echo "Done!"
