import gi
from gi.repository import Gst, GLib
from PIL import Image
import numpy as np

gi.require_version('Gst', '1.0')
Gst.init(None)


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


def create_pipeline(config_path):
    """
    Create a GStreamer pipeline with nvinfer and nvstreammux.
    """
    # Create elements
    pipeline = Gst.Pipeline.new("inference-pipeline")

    appsrc = Gst.ElementFactory.make("appsrc", "source")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    nvstreammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    nvinfer = Gst.ElementFactory.make("nvinfer", "primary-infer")
    fakesink = Gst.ElementFactory.make("fakesink", "sink")

    if not all([pipeline, appsrc, nvvidconv, nvstreammux, nvinfer, fakesink]):
        raise RuntimeError("Failed to create GStreamer elements")

    # Configure nvstreammux
    nvstreammux.set_property("batch-size", 1)
    nvstreammux.set_property("width", 1920)  # Set resolution
    nvstreammux.set_property("height", 1080)
    nvstreammux.set_property("batched-push-timeout", 4000000)

    # Configure nvinfer
    nvinfer.set_property("config-file-path", config_path)

    # Configure appsrc
    appsrc.set_property("caps", Gst.Caps.from_string("video/x-raw, format=RGBA, width=1920, height=1080, framerate=30/1"))
    appsrc.set_property("is-live", True)
    appsrc.set_property("block", True)

    # Link elements
    pipeline.add(appsrc)
    pipeline.add(nvvidconv)
    pipeline.add(nvstreammux)
    pipeline.add(nvinfer)
    pipeline.add(fakesink)

    appsrc.link(nvvidconv)

    # Link nvvidconv to nvstreammux (pad configuration required)
    sinkpad = nvstreammux.get_request_pad("sink_0")
    srcpad = nvvidconv.get_static_pad("src")
    srcpad.link(sinkpad)

    nvstreammux.link(nvinfer)
    nvinfer.link(fakesink)

    # Attach inference result callback to nvinfer
    nvinfer.get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER, inference_callback
    )

    return pipeline, appsrc


def inference_callback(pad, info):
    """
    Callback to retrieve inference results from NvDsBatchMeta.
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

    l_frame = batch_meta.frame_meta_list
    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_obj = frame_meta.obj_meta_list
        while l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            # Retrieve object detection results
                # Extract object information
            instance_id = obj_meta.object_id
            class_id = obj_meta.class_id
            class_name = obj_meta.obj_label

            print(f"Object ID: {instance_id}, Class ID: {class_id}, Class Name: {class_name}")

        l_obj = l_obj.next
        l_frame = l_frame.next

    return Gst.PadProbeReturn.OK


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


# Main function
if __name__ == "__main__":
    import pyds

    # Path to nvinfer configuration file
    config_file_path = "dstest2_pgie_config.txt"

    # Example single image (PIL image)
    image = Image.open("/home/byzkrovnyi/Downloads/test_car.jpg").convert("RGB")

    pipeline, appsrc = create_pipeline(config_file_path)
    push_single_image_to_pipeline(pipeline, appsrc, image)
