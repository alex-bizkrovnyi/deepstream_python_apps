#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import sys

import numpy as np
from PIL import Image

sys.path.append('../')
import platform
import configparser

import gi

gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from common.platform_info import PlatformInfo
from common.bus_call import bus_call

import pyds

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3
MUXER_BATCH_TIMEOUT_USEC = 33000

def inference_callback(pad, info):
    """
    Callback to retrieve inference, tracker, and lost objects information.
    """
    buffer = info.get_buffer()
    if not buffer:
        print("Unable to get Gst.Buffer")
        return Gst.PadProbeReturn.OK

    # Retrieve metadata from the buffer
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
    if not batch_meta:
        print("Unable to get NvDsBatchMeta")
        return Gst.PadProbeReturn.OK

    # Iterate through frames in the batch
    l_frame = batch_meta.frame_meta_list
    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        print(f"Processing Frame {frame_meta.frame_num}")

        # Iterate through objects in the frame
        l_obj = frame_meta.obj_meta_list
        while l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            # Extract object and tracker information
            object_id = obj_meta.object_id  # Unique tracker ID
            class_id = obj_meta.class_id  # Class ID from inference
            class_name = obj_meta.obj_label  # Class label
            tracker_confidence = obj_meta.tracker_confidence  # Tracker confidence

            print(
                f"Object ID: {object_id}, Class ID: {class_id}, "
                f"Class Name: {class_name}, Tracker Confidence: {tracker_confidence}"
            )

            l_obj = l_obj.next

        # Retrieve metadata for lost objects
        l_user = frame_meta.frame_user_meta_list
        while l_user:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDS_TRACKER_PAST_FRAME_META:
                    past_frame_meta = pyds.NvDsPastFrameObjBatch.cast(user_meta.user_meta_data)
                    for i in range(past_frame_meta.numFilled):
                        past_frame_data = pyds.NvDsPastFrameObjList.cast(past_frame_meta.list[i])
                        print(f"Lost Object ID: {past_frame_data.uniqueId}")
            except StopIteration:
                break
            l_user = l_user.next

        l_frame = l_frame.next

    return Gst.PadProbeReturn.OK



# def osd_sink_pad_buffer_probe(pad, info, u_data):
#     frame_number = 0
#     # Intiallizing object counter with 0.
#     obj_counter = {
#         PGIE_CLASS_ID_VEHICLE: 0,
#         PGIE_CLASS_ID_PERSON: 0,
#         PGIE_CLASS_ID_BICYCLE: 0,
#         PGIE_CLASS_ID_ROADSIGN: 0
#     }
#     num_rects = 0
#     gst_buffer = info.get_buffer()
#     if not gst_buffer:
#         print("Unable to get GstBuffer ")
#         return
#
#     # Retrieve batch metadata from the gst_buffer
#     # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
#     # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
#     batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
#     l_frame = batch_meta.frame_meta_list
#     while l_frame is not None:
#         try:
#             # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
#             # The casting is done by pyds.NvDsFrameMeta.cast()
#             # The casting also keeps ownership of the underlying memory
#             # in the C code, so the Python garbage collector will leave
#             # it alone.
#             frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
#         except StopIteration:
#             break
#
#         frame_number = frame_meta.frame_num
#         num_rects = frame_meta.num_obj_meta
#         l_obj = frame_meta.obj_meta_list
#         while l_obj is not None:
#             try:
#                 # Casting l_obj.data to pyds.NvDsObjectMeta
#                 obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
#                 instance_id = obj_meta.object_id
#                 class_id = obj_meta.class_id
#                 class_name = obj_meta.obj_label
#
#                 print(f"Object ID: {instance_id}, Class ID: {class_id}, Class Name: {class_name}")
#             except StopIteration:
#                 break
#             obj_counter[obj_meta.class_id] += 1
#             try:
#                 l_obj = l_obj.next
#             except StopIteration:
#                 break
#
#         # Acquiring a display meta object. The memory ownership remains in
#         # the C code so downstream plugins can still access it. Otherwise
#         # the garbage collector will claim it when this probe function exits.
#         display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
#         display_meta.num_labels = 1
#         py_nvosd_text_params = display_meta.text_params[0]
#         # Setting display text to be shown on screen
#         # Note that the pyds module allocates a buffer for the string, and the
#         # memory will not be claimed by the garbage collector.
#         # Reading the display_text field here will return the C address of the
#         # allocated string. Use pyds.get_string() to get the string content.
#         py_nvosd_text_params.display_text = "Frame Number={} Number of Objects={} Vehicle_count={} Person_count={}".format(
#             frame_number, num_rects, obj_counter[PGIE_CLASS_ID_VEHICLE], obj_counter[PGIE_CLASS_ID_PERSON])
#
#         # Now set the offsets where the string should appear
#         py_nvosd_text_params.x_offset = 10
#         py_nvosd_text_params.y_offset = 12
#
#         # Font , font-color and font-size
#         py_nvosd_text_params.font_params.font_name = "Serif"
#         py_nvosd_text_params.font_params.font_size = 10
#         # set(red, green, blue, alpha); set to White
#         py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
#
#         # Text background color
#         py_nvosd_text_params.set_bg_clr = 1
#         # set(red, green, blue, alpha); set to Black
#         py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
#         # Using pyds.get_string() to get display_text as string
#         print(pyds.get_string(py_nvosd_text_params.display_text))
#         pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
#         try:
#             l_frame = l_frame.next
#         except StopIteration:
#             break
#     # past tracking meta data
#     l_user = batch_meta.batch_user_meta_list
#     while l_user is not None:
#         try:
#             # Note that l_user.data needs a cast to pyds.NvDsUserMeta
#             # The casting is done by pyds.NvDsUserMeta.cast()
#             # The casting also keeps ownership of the underlying memory
#             # in the C code, so the Python garbage collector will leave
#             # it alone
#             user_meta = pyds.NvDsUserMeta.cast(l_user.data)
#         except StopIteration:
#             break
#         if (user_meta and user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDS_TRACKER_PAST_FRAME_META):
#             try:
#                 # Note that user_meta.user_meta_data needs a cast to pyds.NvDsTargetMiscDataBatch
#                 # The casting is done by pyds.NvDsTargetMiscDataBatch.cast()
#                 # The casting also keeps ownership of the underlying memory
#                 # in the C code, so the Python garbage collector will leave
#                 # it alone
#                 pPastDataBatch = pyds.NvDsTargetMiscDataBatch.cast(user_meta.user_meta_data)
#             except StopIteration:
#                 break
#             for miscDataStream in pyds.NvDsTargetMiscDataBatch.list(pPastDataBatch):
#                 print("streamId=", miscDataStream.streamID)
#                 print("surfaceStreamID=", miscDataStream.surfaceStreamID)
#                 for miscDataObj in pyds.NvDsTargetMiscDataStream.list(miscDataStream):
#                     print("numobj=", miscDataObj.numObj)
#                     print("uniqueId=", miscDataObj.uniqueId)
#                     print("classId=", miscDataObj.classId)
#                     print("objLabel=", miscDataObj.objLabel)
#                     for miscDataFrame in pyds.NvDsTargetMiscDataObject.list(miscDataObj):
#                         print('frameNum:', miscDataFrame.frameNum)
#                         print('tBbox.left:', miscDataFrame.tBbox.left)
#                         print('tBbox.width:', miscDataFrame.tBbox.width)
#                         print('tBbox.top:', miscDataFrame.tBbox.top)
#                         print('tBbox.right:', miscDataFrame.tBbox.height)
#                         print('confidence:', miscDataFrame.confidence)
#                         print('age:', miscDataFrame.age)
#         try:
#             l_user = l_user.next
#         except StopIteration:
#             break
#     return Gst.PadProbeReturn.OK


def create_pipeline():
    platform_info = PlatformInfo()
    # Standard GStreamer initialization

    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    # Source element for reading from the file
    print("Creating Source \n ")
    appsrc = Gst.ElementFactory.make("appsrc", "source")
    appsrc.set_property("caps",
                        Gst.Caps.from_string("video/x-raw, format=RGBA, width=1920, height=1080, framerate=30/1"))
    appsrc.set_property("is-live", True)
    appsrc.set_property("block", True)

    # Since the data format in the input file is elementary h264 stream,
    # we need a h264parser
    print("Creating H264Parser \n")
    h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    if not h264parser:
        sys.stderr.write(" Unable to create h264 parser \n")

    # Use nvdec_h264 for hardware accelerated decode on GPU
    print("Creating Decoder \n")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    if not decoder:
        sys.stderr.write(" Unable to create Nvv4l2 Decoder \n")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    # Use nvinfer to run inferencing on decoder's output,
    # behaviour of inferencing is set through config file
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create tracker \n")

    sgie1 = Gst.ElementFactory.make("nvinfer", "secondary1-nvinference-engine")
    if not sgie1:
        sys.stderr.write(" Unable to make sgie1 \n")

    sgie2 = Gst.ElementFactory.make("nvinfer", "secondary2-nvinference-engine")
    if not sgie2:
        sys.stderr.write(" Unable to make sgie2 \n")

    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

    # Create OSD to draw on the converted RGBA buffer
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")

    # Finally render the osd output
    if platform_info.is_integrated_gpu():
        print("Creating nv3dsink \n")
        sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
        if not sink:
            sys.stderr.write(" Unable to create nv3dsink \n")
    else:
        if platform_info.is_platform_aarch64():
            print("Creating nv3dsink \n")
            sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
        else:
            print("Creating EGLSink \n")
            sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        if not sink:
            sys.stderr.write(" Unable to create egl sink \n")

    # print("Playing file %s " %args[1])
    # source.set_property('location', args[1])
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', MUXER_BATCH_TIMEOUT_USEC)

    # Set properties of pgie and sgie
    pgie.set_property('config-file-path', "dstest2_pgie_config.txt")
    sgie1.set_property('config-file-path', "dstest2_sgie1_config.txt")
    sgie2.set_property('config-file-path', "dstest2_sgie2_config.txt")

    # Set properties of tracker
    config = configparser.ConfigParser()
    config.read('dstest2_tracker_config.txt')
    config.sections()

    for key in config['tracker']:
        if key == 'tracker-width':
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height':
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id':
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file':
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file':
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)

    print("Adding elements to Pipeline \n")
    fakesink = Gst.ElementFactory.make("fakesink", "sink")

    pipeline.add(appsrc)
    pipeline.add(nvvidconv)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(fakesink)

    appsrc.link(nvvidconv)

    # Link nvvidconv to nvstreammux (pad configuration required)
    sinkpad = streammux.get_request_pad("sink_0")
    srcpad = nvvidconv.get_static_pad("src")
    srcpad.link(sinkpad)

    streammux.link(pgie)
    pgie.link(fakesink)

    # Attach inference result callback to nvinfer
    pgie.get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER, inference_callback
    )

    # pipeline.add(appsrc)
    # # pipeline.add(h264parser)
    # # pipeline.add(decoder)
    # pipeline.add(streammux)
    # pipeline.add(pgie)
    # pipeline.add(tracker)
    # # pipeline.add(sgie1)
    # # pipeline.add(sgie2)
    # pipeline.add(nvvidconv)
    # pipeline.add(nvosd)
    # pipeline.add(sink)
    #
    # # we link the elements together
    # # file-source -> h264-parser -> nvh264-decoder ->
    # # nvinfer -> nvvidconv -> nvosd -> video-renderer
    # print("Linking elements in the Pipeline \n")
    # appsrc.link(nvvidconv)
    # # h264parser.link(decoder)
    #
    # sinkpad = streammux.get_request_pad("sink_0")
    # srcpad = nvvidconv.get_static_pad("src")
    # srcpad.link(sinkpad)
    #
    # streammux.link(pgie)
    # pgie.link(sink)
    #
    # # pgie.link(tracker)
    # # tracker.link(sgie1)
    # # sgie1.link(sgie2)
    # # sgie2.link(nvvidconv)
    # # nvvidconv.link(nvosd)
    # # nvosd.link(sink)
    #
    # # create and event loop and feed gstreamer bus mesages to it
    # # loop = GLib.MainLoop()
    # #
    # # bus = pipeline.get_bus()
    # # bus.add_signal_watch()
    # # bus.connect ("message", bus_call, loop)
    #
    # # Lets add probe to get informed of the meta data generated, we add probe to
    # # the sink pad of the osd element, since by that time, the buffer would have
    # # had got all the metadata.
    # osdsinkpad = nvosd.get_static_pad("sink")
    # if not osdsinkpad:
    #     sys.stderr.write(" Unable to get sink pad of nvosd \n")
    # # osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
    # osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, inference_callback, 0)

    return pipeline, appsrc


def pil_to_gst_buffer(pil_image):
    """
    Convert a PIL Image to Gst.Buffer.
    """
    np_image = np.array(pil_image, dtype=np.uint8)
    height, width, channels = np_image.shape

    # Ensure input is RGB
    if channels != 3:
        raise ValueError("Input image must have 3 channels (RGB)")

    # Convert to RGBA for DeepStream
    rgba_image = np.dstack([np_image, np.full((height, width), 255, dtype=np.uint8)])

    # Create Gst.Buffer
    gst_buffer = Gst.Buffer.new_wrapped(rgba_image.tobytes())
    return gst_buffer


def push_single_image_to_pipeline(pipeline, appsrc, pil_image):
    """
    Push a single PIL image to the pipeline.
    """
    pipeline.set_state(Gst.State.PLAYING)

    gst_buffer = pil_to_gst_buffer(pil_image)
    appsrc.emit("push-buffer", gst_buffer)
    print("Pushed buffer to appsrc")

    # End the stream
    appsrc.emit("end-of-stream")
    print("End-of-stream signaled")

    # Wait for pipeline to finish
    bus = pipeline.get_bus()
    msg = bus.timed_pop_filtered(
        Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS
    )

    if msg.type == Gst.MessageType.ERROR:
        err, debug = msg.parse_error()
        print("Error:", err, debug)
    elif msg.type == Gst.MessageType.EOS:
        print("Pipeline finished")

    pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    image = Image.open("/home/byzkrovnyi/Downloads/test_car.jpg").convert("RGB")

    pipeline, appsrc = create_pipeline()
    push_single_image_to_pipeline(pipeline, appsrc, image)
