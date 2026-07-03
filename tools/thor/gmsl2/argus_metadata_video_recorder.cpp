/*
 * Metadata-integrated Libargus video recorder for Thor GMSL2 cameras.
 *
 * This is the production-direction replacement for the pure GStreamer
 * nvarguscamerasrc recorder.  Each camera is owned by Libargus.  For each
 * capture request we enable two OutputStreams:
 *
 *   1. video stream    -> nveglstreamsrc -> nvv4l2h265enc -> matroskamux -> cam_XX.mkv
 *   2. metadata stream -> FrameConsumer -> cam_XX.argus_frame_metadata.csv
 *
 * Because both streams are driven by the same Argus request, the sidecar's
 * encoded_frame_index is intended to correspond to the same capture order as
 * the encoded video stream.  The Python save gate then aligns cameras by
 * SOF TSC and rejects unsynchronized episodes.
 */

#include "ArgusHelpers.h"
#include <Argus/Argus.h>
#include <Argus/Ext/InternalFrameCount.h>
#include <Argus/Ext/SensorTimestampTsc.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <EGLStream/EGLStream.h>
#include <gst/gst.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

using namespace Argus;
using namespace EGLStream;
using namespace ArgusSamples;

namespace {

std::atomic<bool> g_stop_requested(false);

void handle_signal(int) {
    g_stop_requested.store(true);
}

class SimpleEGLDisplay {
public:
    ~SimpleEGLDisplay() { cleanup(); }

    bool initialize() {
        display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);

#ifdef EGL_EXT_platform_device
        if (display_ == EGL_NO_DISPLAY) {
            const char* client_extensions = eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);
            if (client_extensions &&
                std::strstr(client_extensions, "EGL_EXT_client_extensions") &&
                std::strstr(client_extensions, "EGL_EXT_device_base") &&
                std::strstr(client_extensions, "EGL_EXT_platform_base") &&
                std::strstr(client_extensions, "EGL_EXT_platform_device")) {
                PFNEGLQUERYDEVICESEXTPROC egl_query_devices_ext =
                    reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC>(eglGetProcAddress("eglQueryDevicesEXT"));
                PFNEGLGETPLATFORMDISPLAYEXTPROC egl_get_platform_display_ext =
                    reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(eglGetProcAddress("eglGetPlatformDisplayEXT"));
                if (egl_query_devices_ext && egl_get_platform_display_ext) {
                    EGLDeviceEXT device;
                    EGLint num_devices = 0;
                    if (egl_query_devices_ext(1, &device, &num_devices) && num_devices == 1) {
                        display_ = egl_get_platform_display_ext(EGL_PLATFORM_DEVICE_EXT, device, nullptr);
                    }
                }
            }
        }
#endif

        if (display_ == EGL_NO_DISPLAY) {
            std::cerr << "could not get EGL display" << std::endl;
            return false;
        }
        if (!eglInitialize(display_, nullptr, nullptr)) {
            std::cerr << "could not initialize EGL display, eglError=0x"
                      << std::hex << eglGetError() << std::dec << std::endl;
            display_ = EGL_NO_DISPLAY;
            return false;
        }
        const char* extensions = eglQueryString(display_, EGL_EXTENSIONS);
        if (!extensions || !std::strstr(extensions, "EGL_KHR_stream")) {
            std::cerr << "EGL_KHR_stream is not supported" << std::endl;
            cleanup();
            return false;
        }
        return true;
    }

    void cleanup() {
        if (display_ != EGL_NO_DISPLAY) {
            eglTerminate(display_);
            display_ = EGL_NO_DISPLAY;
        }
    }

    EGLDisplay get() const { return display_; }

private:
    EGLDisplay display_ = EGL_NO_DISPLAY;
};

SimpleEGLDisplay g_display;

struct Options {
    std::vector<uint32_t> sids;
    uint32_t frames = 120;
    uint32_t sensor_mode = 0;
    uint32_t fps = 60;
    uint32_t bitrate = 40000000;
    uint32_t iframe_interval = 1;
    uint32_t preset_level = 1;
    uint32_t control_rate = 1;
    bool use_h264 = false;
    bool use_mp4 = false;
    std::string name_prefix = "cam";
    std::string episode_dir = ".";
};

struct FrameMetadata {
    uint64_t encoded_frame_index = 0;
    uint64_t local_frame_number = 0;
    uint64_t sensor_timestamp_ns = 0;
    uint64_t sof_tsc_ns = 0;
    uint64_t eof_tsc_ns = 0;
    uint64_t internal_frame_count = 0;
};

std::vector<uint32_t> parse_sids(const std::string& value) {
    std::vector<uint32_t> result;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            result.push_back(static_cast<uint32_t>(std::strtoul(item.c_str(), nullptr, 10)));
        }
    }
    return result;
}

std::string camera_name(const std::string& prefix, uint32_t sid) {
    char suffix[16];
    std::snprintf(suffix, sizeof(suffix), "_%02u", sid);
    return prefix + std::string(suffix);
}

bool mkdir_p(const std::string& path) {
    if (path.empty()) {
        return false;
    }
    std::string current;
    if (path[0] == '/') {
        current = "/";
    }
    std::stringstream ss(path);
    std::string part;
    while (std::getline(ss, part, '/')) {
        if (part.empty()) {
            continue;
        }
        if (!current.empty() && current[current.size() - 1] != '/') {
            current += "/";
        }
        current += part;
        struct stat st;
        if (stat(current.c_str(), &st) == 0) {
            if (!S_ISDIR(st.st_mode)) {
                std::cerr << "path exists but is not a directory: " << current << std::endl;
                return false;
            }
            continue;
        }
        if (mkdir(current.c_str(), 0775) != 0 && errno != EEXIST) {
            std::cerr << "mkdir failed for " << current << ": " << std::strerror(errno) << std::endl;
            return false;
        }
    }
    return true;
}

bool parse_args(int argc, char** argv, Options* options) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto require_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << std::endl;
                return nullptr;
            }
            return argv[++i];
        };
        if (arg == "--sids") {
            const char* value = require_value("--sids");
            if (!value) return false;
            options->sids = parse_sids(value);
        } else if (arg == "--frames") {
            const char* value = require_value("--frames");
            if (!value) return false;
            options->frames = static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
        } else if (arg == "--sensor-mode") {
            const char* value = require_value("--sensor-mode");
            if (!value) return false;
            options->sensor_mode = static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
        } else if (arg == "--fps") {
            const char* value = require_value("--fps");
            if (!value) return false;
            options->fps = static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
        } else if (arg == "--bitrate") {
            const char* value = require_value("--bitrate");
            if (!value) return false;
            options->bitrate = static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
        } else if (arg == "--iframe-interval") {
            const char* value = require_value("--iframe-interval");
            if (!value) return false;
            options->iframe_interval = static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
        } else if (arg == "--preset-level") {
            const char* value = require_value("--preset-level");
            if (!value) return false;
            options->preset_level = static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
        } else if (arg == "--control-rate") {
            const char* value = require_value("--control-rate");
            if (!value) return false;
            options->control_rate = static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
        } else if (arg == "--codec") {
            const char* value = require_value("--codec");
            if (!value) return false;
            std::string codec(value);
            if (codec == "h264") {
                options->use_h264 = true;
            } else if (codec == "h265") {
                options->use_h264 = false;
            } else {
                std::cerr << "--codec must be h264 or h265" << std::endl;
                return false;
            }
        } else if (arg == "--container") {
            const char* value = require_value("--container");
            if (!value) return false;
            std::string container(value);
            if (container == "mp4") {
                options->use_mp4 = true;
            } else if (container == "mkv") {
                options->use_mp4 = false;
            } else {
                std::cerr << "--container must be mkv or mp4" << std::endl;
                return false;
            }
        } else if (arg == "--episode-dir") {
            const char* value = require_value("--episode-dir");
            if (!value) return false;
            options->episode_dir = value;
        } else if (arg == "--name-prefix") {
            const char* value = require_value("--name-prefix");
            if (!value) return false;
            options->name_prefix = value;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "usage: " << argv[0]
                      << " --sids 6,7 --frames 600 --episode-dir DIR"
                      << " [--fps 60] [--codec h265] [--bitrate 40000000]"
                      << " [--iframe-interval 1] [--container mkv]"
                      << " [--name-prefix cam]"
                      << "\n       --frames 0 records until SIGINT/SIGTERM"
                      << std::endl;
            std::exit(0);
        } else {
            std::cerr << "unknown argument: " << arg << std::endl;
            return false;
        }
    }
    if (options->sids.empty()) {
        std::cerr << "--sids is required" << std::endl;
        return false;
    }
    return true;
}

class GstCameraEncoder {
public:
    GstCameraEncoder() = default;
    ~GstCameraEncoder() { shutdown(); }

    bool initialize(
        EGLStreamKHR egl_stream,
        const Size2D<uint32_t>& resolution,
        const Options& options,
        const std::string& output_path
    ) {
        gst_init(nullptr, nullptr);
        pipeline_ = gst_pipeline_new("argus_metadata_video_pipeline");
        if (!pipeline_) {
            std::cerr << "failed to create GStreamer pipeline" << std::endl;
            return false;
        }

        GstElement* video_source = gst_element_factory_make("nveglstreamsrc", nullptr);
        GstElement* queue = gst_element_factory_make("queue", nullptr);
        GstElement* frame_limiter = gst_element_factory_make("identity", nullptr);
        encoder_ = gst_element_factory_make(options.use_h264 ? "nvv4l2h264enc" : "nvv4l2h265enc", nullptr);
        GstElement* parser = gst_element_factory_make(options.use_h264 ? "h264parse" : "h265parse", nullptr);
        GstElement* muxer = gst_element_factory_make(options.use_mp4 ? "qtmux" : "matroskamux", nullptr);
        GstElement* sink = gst_element_factory_make("filesink", nullptr);
        if (!video_source || !queue || !frame_limiter || !encoder_ || !parser || !muxer || !sink) {
            std::cerr << "failed to create one or more GStreamer elements" << std::endl;
            return false;
        }

        g_object_set(G_OBJECT(video_source), "display", g_display.get(), nullptr);
        g_object_set(G_OBJECT(video_source), "eglstream", egl_stream, nullptr);
        if (options.frames > 0) {
            // identity's eos-after emits EOS before forwarding that numbered
            // buffer on the Thor GStreamer stack, so N encoded frames need a
            // limit of N+1. In signal-controlled mode (frames=0), this branch
            // is intentionally unbounded and stop() sends EOS.
            g_object_set(
                G_OBJECT(frame_limiter),
                "eos-after",
                static_cast<gint>(options.frames + 1),
                nullptr
            );
        }
        g_object_set(G_OBJECT(encoder_), "bitrate", options.bitrate, nullptr);
        g_object_set(G_OBJECT(encoder_), "iframeinterval", options.iframe_interval, nullptr);
        g_object_set(G_OBJECT(encoder_), "idrinterval", options.iframe_interval, nullptr);
        g_object_set(G_OBJECT(encoder_), "preset-level", options.preset_level, nullptr);
        g_object_set(G_OBJECT(encoder_), "control-rate", options.control_rate, nullptr);
        g_object_set(G_OBJECT(encoder_), "insert-sps-pps", TRUE, nullptr);
        g_object_set(G_OBJECT(sink), "location", output_path.c_str(), nullptr);

        gst_bin_add_many(GST_BIN(pipeline_), video_source, queue, frame_limiter, encoder_, parser, muxer, sink, nullptr);

        GstCaps* caps = gst_caps_new_simple(
            "video/x-raw",
            "format", G_TYPE_STRING, "NV12",
            "width", G_TYPE_INT, resolution.width(),
            "height", G_TYPE_INT, resolution.height(),
            "framerate", GST_TYPE_FRACTION, options.fps, 1,
            nullptr
        );
        GstCapsFeatures* features = gst_caps_features_new("memory:NVMM", nullptr);
        gst_caps_set_features(caps, 0, features);
        bool linked = gst_element_link_filtered(video_source, queue, caps);
        gst_caps_unref(caps);
        if (!linked) {
            std::cerr << "failed to link nveglstreamsrc to queue" << std::endl;
            return false;
        }
        if (!gst_element_link_many(queue, frame_limiter, encoder_, parser, muxer, sink, nullptr)) {
            std::cerr << "failed to link encoder pipeline" << std::endl;
            return false;
        }
        return true;
    }

    bool start() {
        if (!pipeline_) return false;
        if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "failed to set GStreamer pipeline PLAYING" << std::endl;
            return false;
        }
        playing_ = true;
        return true;
    }

    bool stop() {
        if (!pipeline_ || !playing_) {
            return true;
        }
        gst_element_send_event(pipeline_, gst_event_new_eos());
        GstBus* bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline_));
        if (!bus) {
            std::cerr << "failed to get pipeline bus" << std::endl;
            return false;
        }
        GstMessage* msg = gst_bus_timed_pop_filtered(
            bus,
            5 * GST_SECOND,
            static_cast<GstMessageType>(GST_MESSAGE_EOS | GST_MESSAGE_ERROR)
        );
        if (msg) {
            if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
                GError* err = nullptr;
                gchar* debug = nullptr;
                gst_message_parse_error(msg, &err, &debug);
                std::cerr << "GStreamer error while stopping: "
                          << (err ? err->message : "unknown") << std::endl;
                if (debug) {
                    std::cerr << debug << std::endl;
                }
                if (err) g_error_free(err);
                if (debug) g_free(debug);
                gst_message_unref(msg);
                gst_object_unref(bus);
                return false;
            }
            gst_message_unref(msg);
        } else {
            std::cerr << "timed out waiting for GStreamer EOS" << std::endl;
        }
        gst_object_unref(bus);

        if (gst_element_set_state(pipeline_, GST_STATE_NULL) == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "failed to set GStreamer pipeline NULL" << std::endl;
            return false;
        }
        playing_ = false;
        return true;
    }

    void shutdown() {
        stop();
        if (pipeline_) {
            gst_object_unref(GST_OBJECT(pipeline_));
            pipeline_ = nullptr;
            encoder_ = nullptr;
        }
    }

private:
    GstElement* pipeline_ = nullptr;
    GstElement* encoder_ = nullptr;
    bool playing_ = false;
};

struct CamCtx {
    uint32_t sid = 0;
    std::string name;
    CameraDevice* camera_device = nullptr;
    SensorMode* sensor_mode = nullptr;
    UniqueObj<CaptureSession> session;
    ICaptureSession* i_session = nullptr;
    UniqueObj<OutputStream> video_stream;
    UniqueObj<OutputStream> metadata_stream;
    IFrameConsumer* i_metadata_consumer = nullptr;
    UniqueObj<FrameConsumer> metadata_consumer;
    UniqueObj<Request> request;
    std::ofstream csv;
    std::unique_ptr<GstCameraEncoder> encoder;
    std::thread metadata_thread;
    std::atomic<bool> ok{true};
    std::atomic<uint64_t> metadata_count{0};
    std::atomic<uint64_t> latest_encoded_frame_index{std::numeric_limits<uint64_t>::max()};
    std::atomic<uint64_t> latest_sof_tsc_ns{0};
};

UniqueObj<OutputStream> create_output_stream(
    ICaptureSession* session,
    SensorMode* sensor_mode,
    bool set_display
) {
    ISensorMode* i_sensor_mode = interface_cast<ISensorMode>(sensor_mode);
    UniqueObj<OutputStreamSettings> settings(
        session->createOutputStreamSettings(STREAM_TYPE_EGL)
    );
    IEGLOutputStreamSettings* i_settings = interface_cast<IEGLOutputStreamSettings>(settings);
    if (!i_settings || !i_sensor_mode) {
        return UniqueObj<OutputStream>();
    }
    i_settings->setPixelFormat(PIXEL_FMT_YCbCr_420_888);
    i_settings->setResolution(i_sensor_mode->getResolution());
    i_settings->setMetadataEnable(true);
    if (set_display) {
        i_settings->setEGLDisplay(g_display.get());
    }
    return UniqueObj<OutputStream>(session->createOutputStream(settings.get()));
}

bool init_camera(ICameraProvider* provider, UniqueObj<CameraProvider>& provider_obj, CamCtx* cam, const Options& options) {
    cam->name = camera_name(options.name_prefix, cam->sid);
    cam->camera_device = ArgusHelpers::getCameraDevice(provider_obj.get(), cam->sid);
    if (!cam->camera_device) {
        std::cerr << cam->name << ": camera device unavailable" << std::endl;
        return false;
    }
    cam->sensor_mode = ArgusHelpers::getSensorMode(cam->camera_device, options.sensor_mode);
    ISensorMode* i_sensor_mode = interface_cast<ISensorMode>(cam->sensor_mode);
    if (!i_sensor_mode) {
        std::cerr << cam->name << ": sensor mode unavailable" << std::endl;
        return false;
    }

    cam->session = UniqueObj<CaptureSession>(provider->createCaptureSession(cam->camera_device));
    cam->i_session = interface_cast<ICaptureSession>(cam->session);
    if (!cam->i_session) {
        std::cerr << cam->name << ": createCaptureSession failed" << std::endl;
        return false;
    }

    cam->video_stream = create_output_stream(cam->i_session, cam->sensor_mode, true);
    IEGLOutputStream* i_video_stream = interface_cast<IEGLOutputStream>(cam->video_stream);
    if (!i_video_stream) {
        std::cerr << cam->name << ": video OutputStream failed" << std::endl;
        return false;
    }

    cam->metadata_stream = create_output_stream(cam->i_session, cam->sensor_mode, false);
    if (!cam->metadata_stream) {
        std::cerr << cam->name << ": metadata OutputStream failed" << std::endl;
        return false;
    }
    cam->metadata_consumer = UniqueObj<FrameConsumer>(FrameConsumer::create(cam->metadata_stream.get()));
    cam->i_metadata_consumer = interface_cast<IFrameConsumer>(cam->metadata_consumer);
    if (!cam->i_metadata_consumer) {
        std::cerr << cam->name << ": metadata FrameConsumer failed" << std::endl;
        return false;
    }

    cam->request = UniqueObj<Request>(cam->i_session->createRequest(CAPTURE_INTENT_VIDEO_RECORD));
    IRequest* i_request = interface_cast<IRequest>(cam->request);
    if (!i_request) {
        std::cerr << cam->name << ": createRequest failed" << std::endl;
        return false;
    }
    if (i_request->enableOutputStream(cam->video_stream.get()) != STATUS_OK ||
        i_request->enableOutputStream(cam->metadata_stream.get()) != STATUS_OK) {
        std::cerr << cam->name << ": enableOutputStream failed" << std::endl;
        return false;
    }
    ISourceSettings* i_source_settings = interface_cast<ISourceSettings>(cam->request);
    if (!i_source_settings) {
        std::cerr << cam->name << ": ISourceSettings unavailable" << std::endl;
        return false;
    }
    i_source_settings->setSensorMode(cam->sensor_mode);
    i_source_settings->setFrameDurationRange(1000000000ULL / options.fps);

    std::string video_path = options.episode_dir + "/" + cam->name + (options.use_mp4 ? ".mp4" : ".mkv");
    cam->encoder.reset(new GstCameraEncoder());
    if (!cam->encoder->initialize(i_video_stream->getEGLStream(), i_sensor_mode->getResolution(), options, video_path)) {
        std::cerr << cam->name << ": encoder initialize failed" << std::endl;
        return false;
    }

    std::string sidecar_path = options.episode_dir + "/" + cam->name + ".argus_frame_metadata.csv";
    cam->csv.open(sidecar_path.c_str(), std::ios::out | std::ios::trunc);
    if (!cam->csv) {
        std::cerr << cam->name << ": cannot open " << sidecar_path << std::endl;
        return false;
    }
    cam->csv << "camera,encoded_frame_index,local_frame_number,sensor_timestamp_ns,"
             << "sof_tsc_ns,eof_tsc_ns,internal_frame_count\n";
    return true;
}

bool acquire_one_metadata(
    CamCtx* cam,
    uint64_t encoded_frame_index,
    FrameMetadata* out,
    uint64_t timeout_ns = TIMEOUT_INFINITE
) {
    Status status = STATUS_OK;
    UniqueObj<Frame> frame(cam->i_metadata_consumer->acquireFrame(timeout_ns, &status));
    if (!frame) {
        if (status == STATUS_TIMEOUT) {
            return false;
        }
        std::cerr << cam->name << ": acquireFrame returned null" << std::endl;
        return false;
    }
    IFrame* i_frame = interface_cast<IFrame>(frame);
    IArgusCaptureMetadata* i_argus_meta = interface_cast<IArgusCaptureMetadata>(frame);
    if (!i_frame || !i_argus_meta) {
        std::cerr << cam->name << ": metadata interface missing" << std::endl;
        return false;
    }
    CaptureMetadata* metadata = i_argus_meta->getMetadata();
    ICaptureMetadata* i_meta = interface_cast<ICaptureMetadata>(metadata);
    Ext::ISensorTimestampTsc* i_tsc = interface_cast<Ext::ISensorTimestampTsc>(metadata);
    Ext::IInternalFrameCount* i_internal = interface_cast<Ext::IInternalFrameCount>(metadata);
    out->encoded_frame_index = encoded_frame_index;
    out->local_frame_number = i_frame->getNumber();
    out->sensor_timestamp_ns = i_meta ? i_meta->getSensorTimestamp() : 0;
    out->sof_tsc_ns = i_tsc ? i_tsc->getSensorSofTimestampTsc() : 0;
    out->eof_tsc_ns = i_tsc ? i_tsc->getSensorEofTimestampTsc() : 0;
    out->internal_frame_count = i_internal ? i_internal->getInternalFrameCount() : 0;
    return true;
}

void metadata_loop(CamCtx* cam, uint32_t frames) {
    const uint64_t frame_timeout_ns = 8ULL * 1000ULL * 1000ULL * 1000ULL;
    const uint64_t drain_timeout_ns = 50ULL * 1000ULL * 1000ULL;
    for (uint64_t i = 0; frames == 0 || i < frames; ++i) {
        FrameMetadata meta;
        uint64_t timeout_ns = g_stop_requested.load() ? drain_timeout_ns : frame_timeout_ns;
        if (!acquire_one_metadata(cam, i, &meta, timeout_ns)) {
            if (frames == 0 && g_stop_requested.load()) {
                return;
            }
            std::cerr << cam->name << ": timed out waiting for frame metadata" << std::endl;
            cam->ok = false;
            return;
        }
        cam->csv << cam->name << ","
                 << meta.encoded_frame_index << ","
                 << meta.local_frame_number << ","
                 << meta.sensor_timestamp_ns << ","
                 << meta.sof_tsc_ns << ","
                 << meta.eof_tsc_ns << ","
                 << meta.internal_frame_count << "\n";
        cam->latest_encoded_frame_index.store(meta.encoded_frame_index);
        cam->latest_sof_tsc_ns.store(meta.sof_tsc_ns);
        cam->metadata_count.fetch_add(1);
    }
    cam->csv.flush();
}

bool wait_for_recording_marker(
    const std::vector<std::unique_ptr<CamCtx>>& cameras,
    uint64_t min_rows,
    uint64_t timeout_ms
) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        bool ready = true;
        for (const auto& cam : cameras) {
            if (cam->metadata_count.load() < min_rows) {
                ready = false;
                break;
            }
        }
        if (ready) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
}

void write_recording_markers(
    const Options& options,
    const std::string& reference_name,
    bool start_ready,
    uint64_t start_encoded_frame_index,
    uint64_t start_sof_tsc_ns,
    bool has_stop,
    uint64_t stop_encoded_frame_index_exclusive,
    uint64_t stop_sof_tsc_ns_exclusive
) {
    std::string marker_path = options.episode_dir + "/argus_recording_markers.json";
    std::ofstream out(marker_path.c_str(), std::ios::out | std::ios::trunc);
    if (!out) {
        std::cerr << "failed to write " << marker_path << std::endl;
        return;
    }
    out << "{\n"
        << "  \"reference_camera\": \"" << reference_name << "\",\n"
        << "  \"start_ready\": " << (start_ready ? "true" : "false") << ",\n"
        << "  \"start_encoded_frame_index\": " << start_encoded_frame_index << ",\n"
        << "  \"start_sof_tsc_ns\": " << start_sof_tsc_ns;
    if (has_stop) {
        out << ",\n"
            << "  \"stop_encoded_frame_index_exclusive\": "
            << stop_encoded_frame_index_exclusive << ",\n"
            << "  \"stop_sof_tsc_ns_exclusive\": "
            << stop_sof_tsc_ns_exclusive;
    }
    out << "\n}\n";
}

uint64_t encoded_marker_value(uint64_t encoded_frame_index) {
    return encoded_frame_index == std::numeric_limits<uint64_t>::max() ? 0 : encoded_frame_index;
}

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    Options options;
    if (!parse_args(argc, argv, &options)) {
        return 2;
    }
    if (!mkdir_p(options.episode_dir)) {
        return 2;
    }

    if (!g_display.initialize()) {
        std::cerr << "failed to initialize EGL display" << std::endl;
        return 2;
    }

    UniqueObj<CameraProvider> provider_obj(CameraProvider::create());
    ICameraProvider* provider = interface_cast<ICameraProvider>(provider_obj);
    if (!provider) {
        std::cerr << "create CameraProvider failed" << std::endl;
        return 2;
    }
    std::cerr << "Argus Version: " << provider->getVersion() << std::endl;

    std::vector<std::unique_ptr<CamCtx>> cameras;
    for (uint32_t sid : options.sids) {
        std::unique_ptr<CamCtx> cam(new CamCtx());
        cam->sid = sid;
        if (!init_camera(provider, provider_obj, cam.get(), options)) {
            return 3;
        }
        cameras.push_back(std::move(cam));
    }

    for (auto& cam : cameras) {
        if (!cam->encoder->start()) {
            return 4;
        }
    }
    for (auto& cam : cameras) {
        cam->metadata_thread = std::thread(metadata_loop, cam.get(), options.frames);
    }
    for (auto& cam : cameras) {
        if (cam->i_session->repeat(cam->request.get()) != STATUS_OK) {
            std::cerr << cam->name << ": repeat request failed" << std::endl;
            g_stop_requested.store(true);
            for (auto& started : cameras) {
                if (started->metadata_thread.joinable()) {
                    started->metadata_thread.join();
                }
            }
            return 4;
        }
    }
    CamCtx* reference = cameras.front().get();
    bool start_ready = wait_for_recording_marker(cameras, 2, 3000);
    uint64_t start_encoded_frame_index =
        encoded_marker_value(reference->latest_encoded_frame_index.load());
    uint64_t start_sof_tsc_ns = reference->latest_sof_tsc_ns.load();
    write_recording_markers(
        options,
        reference->name,
        start_ready,
        start_encoded_frame_index,
        start_sof_tsc_ns,
        false,
        0,
        0
    );
    if (!start_ready) {
        std::cerr << "recording failed: one or more cameras did not deliver startup metadata" << std::endl;
        g_stop_requested.store(true);
        for (auto& cam : cameras) {
            if (cam->i_session) {
                cam->i_session->stopRepeat();
                cam->i_session->waitForIdle();
            }
        }
        for (auto& cam : cameras) {
            if (cam->metadata_thread.joinable()) {
                cam->metadata_thread.join();
            }
            if (cam->csv) {
                cam->csv.flush();
                cam->csv.close();
            }
        }
        cameras.clear();
        provider_obj.reset();
        g_display.cleanup();
        return 6;
    }
    std::cerr << "recording started" << std::endl;

    if (options.frames == 0) {
        while (!g_stop_requested.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        uint64_t stop_encoded_frame_index =
            encoded_marker_value(reference->latest_encoded_frame_index.load());
        uint64_t stop_sof_tsc_ns = reference->latest_sof_tsc_ns.load();
        write_recording_markers(
            options,
            reference->name,
            start_ready,
            start_encoded_frame_index,
            start_sof_tsc_ns,
            stop_sof_tsc_ns > 0,
            stop_encoded_frame_index + 1,
            stop_sof_tsc_ns + 1
        );
        for (auto& cam : cameras) {
            if (cam->i_session) {
                cam->i_session->stopRepeat();
                cam->i_session->waitForIdle();
            }
        }
    }

    for (auto& cam : cameras) {
        if (cam->metadata_thread.joinable()) {
            cam->metadata_thread.join();
        }
    }

    if (options.frames > 0) {
        uint64_t stop_encoded_frame_index =
            encoded_marker_value(reference->latest_encoded_frame_index.load());
        uint64_t stop_sof_tsc_ns = reference->latest_sof_tsc_ns.load();
        write_recording_markers(
            options,
            reference->name,
            start_ready,
            start_encoded_frame_index,
            start_sof_tsc_ns,
            stop_sof_tsc_ns > 0,
            stop_encoded_frame_index + 1,
            stop_sof_tsc_ns + 1
        );
    }

    bool ok = true;
    for (auto& cam : cameras) {
        ok = ok && cam->ok.load();
    }
    if (options.frames > 0) {
        for (auto& cam : cameras) {
            if (cam->i_session) {
                cam->i_session->stopRepeat();
                cam->i_session->waitForIdle();
            }
        }
    }
    for (auto& cam : cameras) {
        if (!cam->encoder->stop()) {
            ok = false;
        }
        if (cam->csv) {
            cam->csv.flush();
            cam->csv.close();
        }
    }

    cameras.clear();
    provider_obj.reset();
    g_display.cleanup();

    if (!ok) {
        std::cerr << "recording failed" << std::endl;
        return 5;
    }
    if (options.frames > 0) {
        std::cerr << "frames captured per camera: " << options.frames << std::endl;
    } else {
        std::cerr << "signal-controlled recording stopped" << std::endl;
    }
    return 0;
}
