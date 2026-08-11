/*
 * Online-synchronized Libargus recorder for Thor GMSL2 cameras.
 *
 * This recorder consumes Argus BufferOutputStream buffers in application code,
 * extracts same-buffer metadata, accepts only full SOF-synchronized clusters,
 * queues those DMABUF buffers into per-camera NvVideoEncoder instances, and
 * muxes the encoded packets through a small GStreamer appsrc pipeline.
 *
 * It is intentionally separate from argus_metadata_video_recorder.cpp, whose
 * video branch is nveglstreamsrc -> encoder and can only be synchronized after
 * recording. Here synchronization happens before encoding.
 */

#include "ArgusHelpers.h"
#include <Argus/Argus.h>
#include <Argus/BufferStream.h>
#include <Argus/Ext/InternalFrameCount.h>
#include <Argus/Ext/SensorTimestampTsc.h>
#include <EGL/egl.h>
#include <NvVideoEncoder.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include "NvBufSurface.h"
#include "nvmmapi/NvNativeBuffer.h"
#include <nvbufsurface.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <linux/videodev2.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

using namespace Argus;
using namespace ArgusSamples;

namespace {

std::atomic<bool> g_stop_requested(false);

void handle_signal(int) {
    g_stop_requested.store(true);
}

struct Options {
    std::vector<uint32_t> sids;
    uint32_t frames = 120;  // 0 = signal controlled
    uint32_t sensor_mode = 0;
    uint32_t fps = 60;
    uint32_t bitrate = 40000000;
    uint32_t iframe_interval = 60;
    uint32_t preset_level = 1;
    uint32_t control_rate = 1;
    uint32_t startup_full_clusters = 60;
    uint32_t startup_timeout_ms = 15000;
    uint32_t frame_timeout_ms = 1000;
    uint64_t tolerance_ns = 1000000;
    bool use_h264 = false;
    bool use_mp4 = false;
    std::string missing_frame_policy = "fail_episode";
    std::string stop_mode = "full_cluster";
    std::string name_prefix = "cam";
    std::string episode_dir = ".";
    std::string frame_bus_dir;
    uint32_t frame_bus_every_n = 1;
    std::string preview_frame_bus_dir;
    uint32_t preview_frame_bus_every_n = 12;
    bool persistent = false;
};

enum class CommandType {
    Start,
    Stop,
    PreviewOn,
    PreviewOff,
    Quit,
};

struct Command {
    CommandType type = CommandType::Quit;
    uint32_t idx = 0;
    uint32_t frames = 0;
    std::string episode_dir;
};

std::mutex g_command_mutex;
std::deque<Command> g_commands;
std::atomic<bool> g_episode_stop_requested(false);
std::atomic<bool> g_quit_requested(false);
std::atomic<bool> g_preview_requested(false);
std::mutex g_frame_bus_mutex;
uint64_t g_frame_bus_success_count = 0;

struct FrameMetadata {
    uint64_t local_frame_number = 0;
    uint64_t sensor_timestamp_ns = 0;
    uint64_t sof_tsc_ns = 0;
    uint64_t eof_tsc_ns = 0;
    uint64_t internal_frame_count = 0;
};

static constexpr uint32_t kArgusBuffers = 12;
static EGLDisplay g_egl_display = EGL_NO_DISPLAY;

class DmaBuffer : public NvNativeBuffer, public NvBuffer {
public:
    static DmaBuffer* create(
        const Size2D<uint32_t>& size,
        NvBufSurfaceColorFormat color_format,
        NvBufSurfaceLayout layout
    ) {
        DmaBuffer* buffer = new DmaBuffer(size);
        if (!buffer) {
            return nullptr;
        }

        NvBufSurf::NvCommonAllocateParams params;
        std::memset(&params, 0, sizeof(params));
        params.memtag = NvBufSurfaceTag_CAMERA;
        params.width = size.width();
        params.height = size.height();
        params.colorFormat = color_format;
        params.layout = layout;
        params.memType = NVBUF_MEM_SURFACE_ARRAY;

        if (NvBufSurf::NvAllocate(&params, 1, &buffer->m_fd) != 0) {
            delete buffer;
            return nullptr;
        }
        buffer->planes[0].fd = buffer->m_fd;
        buffer->planes[0].bytesused = 1;
        return buffer;
    }


    ~DmaBuffer() {
        if (m_fd >= 0) {
            NvBufSurf::NvDestroy(m_fd);
            m_fd = -1;
        }
    }

    int get_fd() const { return m_fd; }
    void set_argus_buffer(Buffer* buffer, IBuffer* i_buffer) {
        argus_buffer_ = buffer;
        argus_i_buffer_ = i_buffer;
    }
    Buffer* get_argus_buffer() const { return argus_buffer_; }
    IBuffer* get_argus_i_buffer() const { return argus_i_buffer_; }
    const ICaptureMetadata* get_capture_metadata_interface(const CaptureMetadata* metadata) {
        refresh_metadata_interfaces(metadata);
        return i_capture_metadata_;
    }
    const Ext::ISensorTimestampTsc* get_sensor_timestamp_tsc_interface(const CaptureMetadata* metadata) {
        refresh_metadata_interfaces(metadata);
        return i_sensor_timestamp_tsc_;
    }
    const Ext::IInternalFrameCount* get_internal_frame_count_interface(const CaptureMetadata* metadata) {
        refresh_metadata_interfaces(metadata);
        return i_internal_frame_count_;
    }

private:
    explicit DmaBuffer(const Size2D<uint32_t>& size)
        : NvNativeBuffer(size), NvBuffer(0, 0), argus_buffer_(nullptr), argus_i_buffer_(nullptr),
          cached_metadata_(nullptr), i_capture_metadata_(nullptr), i_sensor_timestamp_tsc_(nullptr),
          i_internal_frame_count_(nullptr) {}

    void refresh_metadata_interfaces(const CaptureMetadata* metadata) {
        if (metadata == cached_metadata_) {
            return;
        }
        cached_metadata_ = metadata;
        i_capture_metadata_ = interface_cast<const ICaptureMetadata>(metadata);
        i_sensor_timestamp_tsc_ = interface_cast<const Ext::ISensorTimestampTsc>(metadata);
        i_internal_frame_count_ = interface_cast<const Ext::IInternalFrameCount>(metadata);
    }

    Buffer* argus_buffer_;
    IBuffer* argus_i_buffer_;
    const CaptureMetadata* cached_metadata_;
    const ICaptureMetadata* i_capture_metadata_;
    const Ext::ISensorTimestampTsc* i_sensor_timestamp_tsc_;
    const Ext::IInternalFrameCount* i_internal_frame_count_;
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

std::string path_join(const std::string& dir, const std::string& name) {
    if (dir.empty()) {
        return name;
    }
    if (dir[dir.size() - 1] == '/') {
        return dir + name;
    }
    return dir + "/" + name;
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
        } else if (arg == "--tolerance-ms") {
            const char* value = require_value("--tolerance-ms");
            if (!value) return false;
            options->tolerance_ns = static_cast<uint64_t>(std::llround(std::atof(value) * 1000000.0));
        } else if (arg == "--startup-full-clusters") {
            const char* value = require_value("--startup-full-clusters");
            if (!value) return false;
            options->startup_full_clusters = static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
        } else if (arg == "--startup-timeout-ms") {
            const char* value = require_value("--startup-timeout-ms");
            if (!value) return false;
            options->startup_timeout_ms = static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
        } else if (arg == "--frame-timeout-ms") {
            const char* value = require_value("--frame-timeout-ms");
            if (!value) return false;
            options->frame_timeout_ms = static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
            if (options->frame_timeout_ms == 0) {
                std::cerr << "--frame-timeout-ms must be > 0" << std::endl;
                return false;
            }
        } else if (arg == "--missing-frame-policy") {
            const char* value = require_value("--missing-frame-policy");
            if (!value) return false;
            options->missing_frame_policy = value;
            if (options->missing_frame_policy != "fail_episode") {
                std::cerr << "--missing-frame-policy currently supports only fail_episode" << std::endl;
                return false;
            }
        } else if (arg == "--stop-mode") {
            const char* value = require_value("--stop-mode");
            if (!value) return false;
            options->stop_mode = value;
            if (options->stop_mode != "full_cluster") {
                std::cerr << "--stop-mode currently supports only full_cluster" << std::endl;
                return false;
            }
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
        } else if (arg == "--frame-bus-dir") {
            const char* value = require_value("--frame-bus-dir");
            if (!value) return false;
            options->frame_bus_dir = value;
        } else if (arg == "--frame-bus-every-n") {
            const char* value = require_value("--frame-bus-every-n");
            if (!value) return false;
            options->frame_bus_every_n = static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
            if (options->frame_bus_every_n == 0) {
                std::cerr << "--frame-bus-every-n must be > 0" << std::endl;
                return false;
            }
        } else if (arg == "--preview-frame-bus-dir") {
            const char* value = require_value("--preview-frame-bus-dir");
            if (!value) return false;
            options->preview_frame_bus_dir = value;
        } else if (arg == "--preview-frame-bus-every-n") {
            const char* value = require_value("--preview-frame-bus-every-n");
            if (!value) return false;
            options->preview_frame_bus_every_n = static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
            if (options->preview_frame_bus_every_n == 0) {
                std::cerr << "--preview-frame-bus-every-n must be > 0" << std::endl;
                return false;
            }
        } else if (arg == "--persistent") {
            options->persistent = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "usage: " << argv[0]
                      << " --sids 6,7 --frames 600 --episode-dir DIR"
                      << " [--fps 60] [--codec h265] [--bitrate 40000000]"
                      << " [--iframe-interval 60] [--container mkv]"
                      << " [--tolerance-ms 1.0] [--startup-full-clusters 60]"
                      << " [--frame-timeout-ms 1000]"
                      << " [--frame-bus-dir /dev/shm/lerobot_online_sync]"
                      << " [--frame-bus-every-n 1]"
                      << " [--preview-frame-bus-dir /dev/shm/lerobot_online_sync_preview]"
                      << " [--preview-frame-bus-every-n 12]"
                      << " [--name-prefix cam]"
                      << " [--persistent]"
                      << "\n       --frames 0 records until SIGINT/SIGTERM"
                      << "\n       --persistent keeps Argus streams open and reads stdin commands:"
                      << "\n           START <idx> <frames> <episode_dir>"
                      << "\n           STOP"
                      << "\n           PREVIEW_ON"
                      << "\n           PREVIEW_OFF"
                      << "\n           QUIT"
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
    if (options->fps == 0) {
        std::cerr << "--fps must be > 0" << std::endl;
        return false;
    }
    return true;
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (unsigned char ch : value) {
        switch (ch) {
            case '"':
                out << "\\\"";
                break;
            case '\\':
                out << "\\\\";
                break;
            case '\b':
                out << "\\b";
                break;
            case '\f':
                out << "\\f";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if (ch < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", ch);
                    out << buf;
                } else {
                    out << static_cast<char>(ch);
                }
        }
    }
    return out.str();
}

class EncodedPacketMuxer {
public:
    EncodedPacketMuxer() = default;
    ~EncodedPacketMuxer() { shutdown(); }

    bool initialize(const Options& options, const std::string& output_path) {
        fps_ = options.fps;
        gst_init(nullptr, nullptr);
        pipeline_ = gst_pipeline_new("argus_online_sync_encoded_mux_pipeline");
        appsrc_ = gst_element_factory_make("appsrc", nullptr);
        GstElement* queue = gst_element_factory_make("queue", nullptr);
        GstElement* parser = gst_element_factory_make(options.use_h264 ? "h264parse" : "h265parse", nullptr);
        GstElement* muxer = gst_element_factory_make(options.use_mp4 ? "qtmux" : "matroskamux", nullptr);
        GstElement* sink = gst_element_factory_make("filesink", nullptr);
        if (!pipeline_ || !appsrc_ || !queue || !parser || !muxer || !sink) {
            std::cerr << "failed to create one or more encoded-mux GStreamer elements" << std::endl;
            return false;
        }

        GstCaps* app_caps = gst_caps_new_simple(
            options.use_h264 ? "video/x-h264" : "video/x-h265",
            "stream-format", G_TYPE_STRING, "byte-stream",
            "alignment", G_TYPE_STRING, "au",
            "framerate", GST_TYPE_FRACTION, static_cast<int>(fps_), 1,
            nullptr
        );
        g_object_set(G_OBJECT(appsrc_),
                     "caps", app_caps,
                     "is-live", TRUE,
                     "format", GST_FORMAT_TIME,
                     "block", TRUE,
                     "do-timestamp", FALSE,
                     nullptr);
        gst_caps_unref(app_caps);

        g_object_set(G_OBJECT(sink), "location", output_path.c_str(), nullptr);

        gst_bin_add_many(GST_BIN(pipeline_), appsrc_, queue, parser, muxer, sink, nullptr);
        if (!gst_element_link_many(appsrc_, queue, parser, muxer, sink, nullptr)) {
            std::cerr << "failed to link encoded-mux pipeline" << std::endl;
            return false;
        }
        return true;
    }

    bool start() {
        if (!pipeline_) return false;
        if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "failed to set online-sync GStreamer pipeline PLAYING" << std::endl;
            return false;
        }
        playing_ = true;
        return true;
    }

    bool push_packet(const uint8_t* data, size_t size, uint64_t pts_ns, uint64_t duration_ns) {
        if (!appsrc_ || !playing_) return false;
        if (!data || size == 0) {
            return false;
        }
        GstBuffer* buffer = gst_buffer_new_allocate(nullptr, size, nullptr);
        if (!buffer) {
            std::cerr << "gst_buffer_new_allocate failed for encoded packet" << std::endl;
            return false;
        }
        GstMapInfo map;
        if (!gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
            std::cerr << "gst_buffer_map failed for encoded packet" << std::endl;
            gst_buffer_unref(buffer);
            return false;
        }
        std::memcpy(map.data, data, size);
        gst_buffer_unmap(buffer, &map);
        GST_BUFFER_PTS(buffer) = static_cast<GstClockTime>(pts_ns);
        GST_BUFFER_DTS(buffer) = GST_BUFFER_PTS(buffer);
        GST_BUFFER_DURATION(buffer) = static_cast<GstClockTime>(duration_ns);
        GstFlowReturn flow = gst_app_src_push_buffer(GST_APP_SRC(appsrc_), buffer);
        if (flow != GST_FLOW_OK) {
            std::cerr << "gst_app_src_push_buffer failed: " << flow << std::endl;
            return false;
        }
        return true;
    }

    bool stop() {
        if (!pipeline_ || !playing_) {
            return true;
        }
        GstFlowReturn flow = gst_app_src_end_of_stream(GST_APP_SRC(appsrc_));
        if (flow != GST_FLOW_OK && flow != GST_FLOW_FLUSHING) {
            std::cerr << "gst_app_src_end_of_stream failed: " << flow << std::endl;
        }
        GstBus* bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline_));
        if (!bus) {
            std::cerr << "failed to get online-sync pipeline bus" << std::endl;
            return false;
        }
        GstMessage* msg = gst_bus_timed_pop_filtered(
            bus,
            10 * GST_SECOND,
            static_cast<GstMessageType>(GST_MESSAGE_EOS | GST_MESSAGE_ERROR)
        );
        bool ok = true;
        if (msg) {
            if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
                GError* err = nullptr;
                gchar* debug = nullptr;
                gst_message_parse_error(msg, &err, &debug);
                std::cerr << "GStreamer error while stopping online-sync encoder: "
                          << (err ? err->message : "unknown") << std::endl;
                if (debug) std::cerr << debug << std::endl;
                if (err) g_error_free(err);
                if (debug) g_free(debug);
                ok = false;
            }
            gst_message_unref(msg);
        } else {
            std::cerr << "timed out waiting for online-sync GStreamer EOS" << std::endl;
            ok = false;
        }
        gst_object_unref(bus);
        if (gst_element_set_state(pipeline_, GST_STATE_NULL) == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "failed to set online-sync pipeline NULL" << std::endl;
            ok = false;
        }
        playing_ = false;
        return ok;
    }

    void shutdown() {
        stop();
        if (pipeline_) {
            gst_object_unref(GST_OBJECT(pipeline_));
            pipeline_ = nullptr;
            appsrc_ = nullptr;
        }
    }

private:
    GstElement* pipeline_ = nullptr;
    GstElement* appsrc_ = nullptr;
    uint32_t fps_ = 60;
    bool playing_ = false;
};

class MmapiCameraEncoder {
public:
    MmapiCameraEncoder() = default;
    ~MmapiCameraEncoder() { shutdown(); }

    bool initialize(
        const Size2D<uint32_t>& resolution,
        IBufferOutputStream* argus_stream,
        const Options& options,
        const std::string& output_path,
        const std::string& encoder_name
    ) {
        width_ = resolution.width();
        height_ = resolution.height();
        argus_stream_ = argus_stream;
        fps_ = options.fps;
        frame_duration_ns_ = 1000000000ULL / std::max<uint32_t>(fps_, 1);
        use_h264_ = options.use_h264;

        if (!muxer_.initialize(options, output_path)) {
            return false;
        }

        encoder_.reset(NvVideoEncoder::createVideoEncoder(encoder_name.c_str()));
        if (!encoder_) {
            std::cerr << "Could not create NvVideoEncoder " << encoder_name << std::endl;
            return false;
        }

        uint32_t coded_fmt = use_h264_ ? V4L2_PIX_FMT_H264 : V4L2_PIX_FMT_H265;
        if (encoder_->setCapturePlaneFormat(coded_fmt, width_, height_, 2 * 1024 * 1024) < 0) {
            std::cerr << "Could not set encoder capture plane format" << std::endl;
            return false;
        }
        if (encoder_->setOutputPlaneFormat(V4L2_PIX_FMT_NV12M, width_, height_) < 0) {
            std::cerr << "Could not set encoder output plane format" << std::endl;
            return false;
        }
        if (encoder_->setBitrate(options.bitrate) < 0) {
            std::cerr << "Could not set encoder bitrate" << std::endl;
            return false;
        }
        if (use_h264_) {
            if (encoder_->setProfile(V4L2_MPEG_VIDEO_H264_PROFILE_HIGH) < 0) {
                std::cerr << "Could not set H.264 profile" << std::endl;
                return false;
            }
            if (encoder_->setLevel(V4L2_MPEG_VIDEO_H264_LEVEL_5_1) < 0) {
                std::cerr << "Could not set H.264 level" << std::endl;
                return false;
            }
        } else {
            if (encoder_->setProfile(V4L2_MPEG_VIDEO_H265_PROFILE_MAIN) < 0) {
                std::cerr << "Could not set H.265 profile" << std::endl;
                return false;
            }
        }
        if (encoder_->setRateControlMode(V4L2_MPEG_VIDEO_BITRATE_MODE_CBR) < 0) {
            std::cerr << "Could not set encoder rate control mode" << std::endl;
            return false;
        }
        if (encoder_->setIFrameInterval(options.iframe_interval) < 0) {
            std::cerr << "Could not set I-frame interval" << std::endl;
            return false;
        }
        if (encoder_->setIDRInterval(options.iframe_interval) < 0) {
            std::cerr << "Could not set IDR interval" << std::endl;
            return false;
        }
        if (encoder_->setInsertSpsPpsAtIdrEnabled(true) < 0) {
            std::cerr << "Could not enable SPS/PPS at IDR" << std::endl;
            return false;
        }
        if (encoder_->setFrameRate(options.fps, 1) < 0) {
            std::cerr << "Could not set encoder framerate" << std::endl;
            return false;
        }
        if (encoder_->setHWPresetType(V4L2_ENC_HW_PRESET_ULTRAFAST) < 0) {
            std::cerr << "Could not set encoder HW preset" << std::endl;
            return false;
        }

        if (encoder_->output_plane.setupPlane(V4L2_MEMORY_DMABUF, kOutputBuffers, true, false) < 0) {
            std::cerr << "Could not setup encoder output plane" << std::endl;
            return false;
        }
        if (encoder_->capture_plane.setupPlane(V4L2_MEMORY_MMAP, kCaptureBuffers, true, false) < 0) {
            std::cerr << "Could not setup encoder capture plane" << std::endl;
            return false;
        }
        output_argus_buffers_.assign(encoder_->output_plane.getNumBuffers(), nullptr);
        return true;
    }

    bool start() {
        if (!encoder_) return false;
        if (!muxer_.start()) {
            return false;
        }
        if (encoder_->output_plane.setStreamStatus(true) < 0) {
            std::cerr << "Failed to stream on encoder output plane" << std::endl;
            return false;
        }
        if (encoder_->capture_plane.setStreamStatus(true) < 0) {
            std::cerr << "Failed to stream on encoder capture plane" << std::endl;
            return false;
        }
        encoder_->capture_plane.setDQThreadCallback(&MmapiCameraEncoder::capture_plane_callback);
        encoder_->capture_plane.startDQThread(this);

        for (uint32_t i = 0; i < encoder_->capture_plane.getNumBuffers(); ++i) {
            struct v4l2_buffer v4l2_buf;
            struct v4l2_plane planes[MAX_PLANES];
            std::memset(&v4l2_buf, 0, sizeof(v4l2_buf));
            std::memset(planes, 0, sizeof(planes));
            v4l2_buf.index = i;
            v4l2_buf.m.planes = planes;
            if (encoder_->capture_plane.qBuffer(v4l2_buf, nullptr) < 0) {
                std::cerr << "Failed to enqueue encoder capture buffer" << std::endl;
                return false;
            }
        }
        streaming_ = true;
        return true;
    }

    bool push_buffer(Buffer* argus_buffer, DmaBuffer* dma_buffer, uint64_t logical_index) {
        if (!encoder_ || !streaming_) return false;
        if (!argus_stream_ || !argus_buffer || !dma_buffer) {
            std::cerr << "push_buffer missing Argus stream/buffer" << std::endl;
            return false;
        }
        int slot = find_free_output_slot();
        if (slot < 0 && !reclaim_output_slot(kDqRetries, &slot, false)) {
            return false;
        }
        if (slot < 0 || static_cast<size_t>(slot) >= output_argus_buffers_.size()) {
            std::cerr << "Invalid encoder output slot" << std::endl;
            return false;
        }
        if (output_argus_buffers_[slot]) {
            release_slot(slot);
        }

        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
        std::memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        std::memset(planes, 0, sizeof(planes));
        v4l2_buf.index = static_cast<uint32_t>(slot);
        v4l2_buf.m.planes = planes;
        v4l2_buf.flags |= V4L2_BUF_FLAG_TIMESTAMP_COPY;
        uint64_t pts_us = (logical_index * frame_duration_ns_) / 1000ULL;
        v4l2_buf.timestamp.tv_sec = static_cast<time_t>(pts_us / 1000000ULL);
        v4l2_buf.timestamp.tv_usec = static_cast<suseconds_t>(pts_us % 1000000ULL);

        if (encoder_->output_plane.qBuffer(v4l2_buf, dma_buffer) < 0) {
            std::cerr << "Error while queueing encoder output buffer" << std::endl;
            return false;
        }
        output_argus_buffers_[slot] = argus_buffer;
        inflight_count_ += 1;
        if (inflight_count_ >= kMaxInFlightBeforeReclaim) {
            int reclaimed_slot = -1;
            if (!reclaim_output_slot(kDqRetries, &reclaimed_slot, false)) {
                std::cerr << "Failed to reclaim encoder output buffer during steady-state push" << std::endl;
                return false;
            }
        }
        return true;
    }

    bool stop() {
        bool ok = true;
        if (!encoder_) {
            return muxer_.stop();
        }
        if (streaming_) {
            if (!queue_eos_buffer()) {
                ok = false;
            }
            if (encoder_->capture_plane.waitForDQThread(10000) < 0) {
                std::cerr << "Timed out waiting for encoder capture DQ thread" << std::endl;
                ok = false;
            }
            encoder_->output_plane.setStreamStatus(false);
            encoder_->capture_plane.setStreamStatus(false);
            streaming_ = false;
        }
        for (size_t i = 0; i < output_argus_buffers_.size(); ++i) {
            if (output_argus_buffers_[i]) {
                release_slot(static_cast<int>(i));
            }
        }
        inflight_count_ = 0;
        if (!muxer_.stop()) {
            ok = false;
        }
        return ok;
    }

    void shutdown() {
        stop();
        encoder_.reset();
    }

private:
    static constexpr uint32_t kOutputBuffers = 10;
    static constexpr uint32_t kCaptureBuffers = 6;
    static constexpr uint32_t kMaxInFlightBeforeReclaim = 5;
    static constexpr int kDqRetries = 200;

    static bool capture_plane_callback(
        struct v4l2_buffer* v4l2_buf,
        NvBuffer* buffer,
        NvBuffer*,
        void* arg
    ) {
        MmapiCameraEncoder* self = static_cast<MmapiCameraEncoder*>(arg);
        if (!self || !v4l2_buf || !buffer) {
            return false;
        }
        if (buffer->planes[0].bytesused == 0) {
            return false;
        }
        uint64_t pts_ns = (
            static_cast<uint64_t>(v4l2_buf->timestamp.tv_sec) * 1000000000ULL
            + static_cast<uint64_t>(v4l2_buf->timestamp.tv_usec) * 1000ULL
        );
        if (pts_ns == 0) {
            std::lock_guard<std::mutex> lock(self->packet_lock_);
            pts_ns = self->encoded_packet_count_ * self->frame_duration_ns_;
            self->encoded_packet_count_ += 1;
        }
        if (!self->muxer_.push_packet(
                buffer->planes[0].data,
                buffer->planes[0].bytesused,
                pts_ns,
                self->frame_duration_ns_)) {
            return false;
        }
        if (self->encoder_->capture_plane.qBuffer(*v4l2_buf, nullptr) < 0) {
            std::cerr << "Failed to requeue encoder capture buffer" << std::endl;
            return false;
        }
        return true;
    }

    void release_argus_buffer(Buffer* buffer) {
        if (!buffer) {
            return;
        }
        if (argus_stream_->releaseBuffer(buffer) != STATUS_OK) {
            std::cerr << "Failed to release Argus buffer after encoder output DQ" << std::endl;
        }
    }

    void release_slot(int slot) {
        if (slot < 0 || static_cast<size_t>(slot) >= output_argus_buffers_.size()) return;
        Buffer* buffer = output_argus_buffers_[slot];
        if (!buffer) return;
        release_argus_buffer(buffer);
        output_argus_buffers_[slot] = nullptr;
    }

    int find_free_output_slot() const {
        for (size_t i = 0; i < output_argus_buffers_.size(); ++i) {
            if (!output_argus_buffers_[i]) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    bool queue_eos_buffer() {
        int slot = find_free_output_slot();
        if (slot < 0 && !reclaim_output_slot(kDqRetries, &slot, false)) {
            std::cerr << "Failed to get encoder output slot for EOS" << std::endl;
            return false;
        }

        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
        std::memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        std::memset(planes, 0, sizeof(planes));
        v4l2_buf.index = static_cast<uint32_t>(slot);
        v4l2_buf.m.planes = planes;
        v4l2_buf.m.planes[0].m.fd = -1;
        v4l2_buf.m.planes[0].bytesused = 0;
        if (encoder_->output_plane.qBuffer(v4l2_buf, nullptr) < 0) {
            std::cerr << "Failed to queue encoder EOS buffer" << std::endl;
            return false;
        }
        return true;
    }

    bool reclaim_output_slot(int retries, int* slot_out, bool quiet) {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
        NvBuffer* shared_buffer = nullptr;
        std::memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        std::memset(planes, 0, sizeof(planes));
        v4l2_buf.m.planes = planes;
        if (encoder_->output_plane.dqBuffer(v4l2_buf, nullptr, &shared_buffer, retries) < 0) {
            if (!quiet) {
                std::cerr << "ERROR while DQing encoder output buffer" << std::endl;
            }
            return false;
        }
        int slot = static_cast<int>(v4l2_buf.index);
        Buffer* buffer_to_release = nullptr;
        DmaBuffer* dma = static_cast<DmaBuffer*>(shared_buffer);
        if (dma) {
            buffer_to_release = dma->get_argus_buffer();
        }
        if (!buffer_to_release && slot >= 0 && static_cast<size_t>(slot) < output_argus_buffers_.size()) {
            buffer_to_release = output_argus_buffers_[slot];
        }
        release_argus_buffer(buffer_to_release);
        if (slot >= 0 && static_cast<size_t>(slot) < output_argus_buffers_.size()) {
            output_argus_buffers_[slot] = nullptr;
        }
        if (inflight_count_ > 0) {
            inflight_count_ -= 1;
        }
        if (slot_out) {
            *slot_out = slot;
        }
        return true;
    }

    EncodedPacketMuxer muxer_;
    std::unique_ptr<NvVideoEncoder> encoder_;
    IBufferOutputStream* argus_stream_ = nullptr;
    std::vector<Buffer*> output_argus_buffers_;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint32_t fps_ = 60;
    uint64_t frame_duration_ns_ = 16666666;
    bool use_h264_ = false;
    bool streaming_ = false;
    size_t inflight_count_ = 0;
    std::mutex packet_lock_;
    uint64_t encoded_packet_count_ = 0;
};

struct CamCtx {
    ~CamCtx() {
        cleanup_buffers();
    }

    void cleanup_buffers() {
        argus_buffers.clear();
        for (NvBufSurface* surf : surfaces) {
            if (surf) {
                NvBufSurfaceUnMapEglImage(surf, 0);
            }
        }
        surfaces.clear();
        native_buffers.clear();
    }

    uint32_t sid = 0;
    std::string name;
    CameraDevice* camera_device = nullptr;
    SensorMode* sensor_mode = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    UniqueObj<CaptureSession> session;
    ICaptureSession* i_session = nullptr;
    UniqueObj<OutputStream> stream;
    IBufferOutputStream* i_buffer_stream = nullptr;
    std::vector<std::unique_ptr<DmaBuffer>> native_buffers;
    std::vector<NvBufSurface*> surfaces;
    std::vector<UniqueObj<Buffer>> argus_buffers;
    UniqueObj<Request> request;
    std::ofstream csv;
    std::unique_ptr<MmapiCameraEncoder> encoder;
    uint64_t accepted_frames = 0;
    uint64_t max_abs_delta_ns = 0;
    bool ok = true;
};

struct FrameBundle {
    FrameBundle(CamCtx* cam_, Buffer* buffer_, DmaBuffer* dma_buffer_)
        : cam(cam_), buffer(buffer_), dma_buffer(dma_buffer_) {}
    CamCtx* cam = nullptr;
    Buffer* buffer = nullptr;
    DmaBuffer* dma_buffer = nullptr;
    FrameMetadata meta;
};

UniqueObj<OutputStream> create_output_stream(ICaptureSession* session, SensorMode* sensor_mode) {
    (void)sensor_mode;
    UniqueObj<OutputStreamSettings> settings(session->createOutputStreamSettings(STREAM_TYPE_BUFFER));
    IBufferOutputStreamSettings* i_settings = interface_cast<IBufferOutputStreamSettings>(settings);
    if (!i_settings) {
        return UniqueObj<OutputStream>();
    }
    i_settings->setBufferType(BUFFER_TYPE_EGL_IMAGE);
    i_settings->setMetadataEnable(true);
    return UniqueObj<OutputStream>(session->createOutputStream(settings.get()));
}

bool allocate_argus_buffers(CamCtx* cam, const Size2D<uint32_t>& resolution) {
    cam->i_buffer_stream = interface_cast<IBufferOutputStream>(cam->stream);
    if (!cam->i_buffer_stream) {
        std::cerr << cam->name << ": IBufferOutputStream unavailable" << std::endl;
        return false;
    }
    UniqueObj<BufferSettings> buffer_settings(cam->i_buffer_stream->createBufferSettings());
    IEGLImageBufferSettings* i_buffer_settings = interface_cast<IEGLImageBufferSettings>(buffer_settings);
    if (!i_buffer_settings) {
        std::cerr << cam->name << ": IEGLImageBufferSettings unavailable" << std::endl;
        return false;
    }

    NvBufSurfaceLayout layout = NVBUF_LAYOUT_BLOCK_LINEAR;
    if (access("/dev/nvidia0", F_OK) == 0) {
        // NVIDIA's Thor sample uses pitch layout when the GPU device is present.
        layout = NVBUF_LAYOUT_PITCH;
    }

    cam->native_buffers.reserve(kArgusBuffers);
    cam->surfaces.reserve(kArgusBuffers);
    cam->argus_buffers.reserve(kArgusBuffers);
    for (uint32_t i = 0; i < kArgusBuffers; ++i) {
        std::unique_ptr<DmaBuffer> dma(DmaBuffer::create(resolution, NVBUF_COLOR_FORMAT_NV12, layout));
        if (!dma) {
            std::cerr << cam->name << ": failed to allocate DmaBuffer" << std::endl;
            return false;
        }

        NvBufSurface* surf = nullptr;
        if (NvBufSurfaceFromFd(dma->get_fd(), reinterpret_cast<void**>(&surf)) != 0 || !surf) {
            std::cerr << cam->name << ": NvBufSurfaceFromFd failed for DmaBuffer" << std::endl;
            return false;
        }
        if (NvBufSurfaceMapEglImage(surf, 0) != 0) {
            std::cerr << cam->name << ": NvBufSurfaceMapEglImage failed" << std::endl;
            return false;
        }
        EGLImageKHR egl_image = surf->surfaceList[0].mappedAddr.eglImage;
        if (egl_image == EGL_NO_IMAGE_KHR) {
            std::cerr << cam->name << ": mapped EGLImage is invalid" << std::endl;
            return false;
        }

        i_buffer_settings->setEGLImage(egl_image);
        i_buffer_settings->setEGLDisplay(g_egl_display);
        UniqueObj<Buffer> argus_buffer(cam->i_buffer_stream->createBuffer(buffer_settings.get()));
        IBuffer* i_buffer = interface_cast<IBuffer>(argus_buffer);
        if (!argus_buffer || !i_buffer) {
            std::cerr << cam->name << ": failed to create Argus Buffer" << std::endl;
            return false;
        }
        i_buffer->setClientData(dma.get());
        dma->set_argus_buffer(argus_buffer.get(), i_buffer);
        if (cam->i_buffer_stream->releaseBuffer(argus_buffer.get()) != STATUS_OK) {
            std::cerr << cam->name << ": failed to release initial Argus Buffer" << std::endl;
            return false;
        }

        cam->surfaces.push_back(surf);
        cam->native_buffers.push_back(std::move(dma));
        cam->argus_buffers.push_back(std::move(argus_buffer));
    }
    return true;
}

void close_episode_outputs(CamCtx* cam) {
    if (!cam) {
        return;
    }
    if (cam->encoder) {
        cam->encoder->shutdown();
        cam->encoder.reset();
    }
    if (cam->csv) {
        cam->csv.flush();
        cam->csv.close();
    }
}

bool open_episode_outputs(
    CamCtx* cam,
    const Options& options,
    const Size2D<uint32_t>& resolution,
    const std::string& episode_dir
) {
    close_episode_outputs(cam);
    cam->accepted_frames = 0;
    cam->max_abs_delta_ns = 0;
    cam->ok = true;

    std::string video_path = episode_dir + "/" + cam->name + (options.use_mp4 ? ".mp4" : ".mkv");
    cam->encoder.reset(new MmapiCameraEncoder());
    std::string encoder_name = "online_sync_" + cam->name;
    if (!cam->encoder->initialize(
            resolution,
            cam->i_buffer_stream,
            options,
            video_path,
            encoder_name)) {
        std::cerr << cam->name << ": encoder initialize failed" << std::endl;
        return false;
    }

    std::string sidecar_path = episode_dir + "/" + cam->name + ".argus_frame_metadata.csv";
    cam->csv.open(sidecar_path.c_str(), std::ios::out | std::ios::trunc);
    if (!cam->csv) {
        std::cerr << cam->name << ": cannot open " << sidecar_path << std::endl;
        return false;
    }
    cam->csv << "camera,logical_frame_index,local_frame_number,sensor_timestamp_ns,"
             << "sof_tsc_ns,eof_tsc_ns,internal_frame_count\n";
    return true;
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
    const Size2D<uint32_t> resolution = i_sensor_mode->getResolution();
    cam->width = resolution.width();
    cam->height = resolution.height();
    cam->session = UniqueObj<CaptureSession>(provider->createCaptureSession(cam->camera_device));
    cam->i_session = interface_cast<ICaptureSession>(cam->session);
    if (!cam->i_session) {
        std::cerr << cam->name << ": createCaptureSession failed" << std::endl;
        return false;
    }
    cam->stream = create_output_stream(cam->i_session, cam->sensor_mode);
    if (!cam->stream) {
        std::cerr << cam->name << ": OutputStream failed" << std::endl;
        return false;
    }
    if (!allocate_argus_buffers(cam, resolution)) {
        return false;
    }
    cam->request = UniqueObj<Request>(cam->i_session->createRequest(CAPTURE_INTENT_VIDEO_RECORD));
    IRequest* i_request = interface_cast<IRequest>(cam->request);
    if (!i_request) {
        std::cerr << cam->name << ": createRequest failed" << std::endl;
        return false;
    }
    if (i_request->enableOutputStream(cam->stream.get()) != STATUS_OK) {
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
    return true;
}

DmaBuffer* find_dma_for_argus_buffer(CamCtx* cam, Buffer* buffer) {
    if (!cam || !buffer) {
        return nullptr;
    }
    for (const auto& dma : cam->native_buffers) {
        if (dma && dma->get_argus_buffer() == buffer) {
            return dma.get();
        }
    }
    return nullptr;
}

bool extract_metadata(DmaBuffer* dma, FrameMetadata* out) {
    if (!dma || !dma->get_argus_i_buffer()) {
        return false;
    }
    const CaptureMetadata* metadata = dma->get_argus_i_buffer()->getMetadata();
    const ICaptureMetadata* i_meta = dma->get_capture_metadata_interface(metadata);
    const Ext::ISensorTimestampTsc* i_tsc = dma->get_sensor_timestamp_tsc_interface(metadata);
    const Ext::IInternalFrameCount* i_internal = dma->get_internal_frame_count_interface(metadata);
    out->local_frame_number = i_meta ? i_meta->getCaptureId() : 0;
    out->sensor_timestamp_ns = i_meta ? i_meta->getSensorTimestamp() : 0;
    out->sof_tsc_ns = i_tsc ? i_tsc->getSensorSofTimestampTsc() : 0;
    out->eof_tsc_ns = i_tsc ? i_tsc->getSensorEofTimestampTsc() : 0;
    out->internal_frame_count = i_internal ? i_internal->getInternalFrameCount() : 0;
    return out->sof_tsc_ns > 0;
}

std::unique_ptr<FrameBundle> acquire_one(
    CamCtx* cam,
    uint64_t timeout_ns,
    std::string* failure_detail
) {
    Status status = STATUS_OK;
    Buffer* raw = cam->i_buffer_stream->acquireBuffer(timeout_ns, &status);
    if (!raw) {
        std::ostringstream oss;
        if (status == STATUS_TIMEOUT) {
            oss << cam->name << ": timed out waiting for Argus buffer after "
                << (timeout_ns / 1000000ULL) << " ms";
        } else {
            oss << cam->name << ": acquireBuffer failed, status=" << status;
        }
        if (failure_detail) {
            *failure_detail = oss.str();
        }
        std::cerr << oss.str() << std::endl;
        return nullptr;
    }
    DmaBuffer* dma = find_dma_for_argus_buffer(cam, raw);
    if (!dma) {
        std::ostringstream oss;
        oss << cam->name << ": acquired Argus Buffer without matching cached DmaBuffer";
        if (failure_detail) {
            *failure_detail = oss.str();
        }
        std::cerr << oss.str() << std::endl;
        cam->i_buffer_stream->releaseBuffer(raw);
        return nullptr;
    }
    std::unique_ptr<FrameBundle> bundle(new FrameBundle(cam, raw, dma));
    if (!extract_metadata(dma, &bundle->meta)) {
        std::ostringstream oss;
        oss << cam->name << ": failed to extract same-frame metadata";
        if (failure_detail) {
            *failure_detail = oss.str();
        }
        std::cerr << oss.str() << std::endl;
        cam->i_buffer_stream->releaseBuffer(raw);
        return nullptr;
    }
    return bundle;
}

void release_bundle(FrameBundle* bundle) {
    if (!bundle || !bundle->buffer) {
        return;
    }
    if (bundle->cam && bundle->cam->i_buffer_stream) {
        bundle->cam->i_buffer_stream->releaseBuffer(bundle->buffer);
    }
    bundle->buffer = nullptr;
    bundle->dma_buffer = nullptr;
}

void release_cluster(std::vector<std::unique_ptr<FrameBundle>>* cluster) {
    if (!cluster) {
        return;
    }
    for (auto& bundle : *cluster) {
        release_bundle(bundle.get());
    }
    cluster->clear();
}

bool write_text_atomic(const std::string& path, const std::string& content) {
    const std::string tmp_path = path + ".tmp";
    {
        std::ofstream out(tmp_path.c_str(), std::ios::out | std::ios::trunc);
        if (!out) {
            std::cerr << "frame bus: failed to open " << tmp_path << std::endl;
            return false;
        }
        out << content;
        out.flush();
        if (!out) {
            std::cerr << "frame bus: failed to write " << tmp_path << std::endl;
            return false;
        }
    }
    if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
        std::cerr << "frame bus: failed to rename " << tmp_path
                  << " to " << path << ": " << std::strerror(errno) << std::endl;
        return false;
    }
    return true;
}

bool copy_nv12_to_file(DmaBuffer* dma, uint32_t width, uint32_t height, const std::string& path) {
    if (!dma || width == 0 || height == 0) {
        return false;
    }
    NvBufSurface* surf = nullptr;
    if (NvBufSurfaceFromFd(dma->get_fd(), reinterpret_cast<void**>(&surf)) != 0 || !surf) {
        std::cerr << "frame bus: NvBufSurfaceFromFd failed" << std::endl;
        return false;
    }
    if (NvBufSurfaceMap(surf, 0, -1, NVBUF_MAP_READ) != 0) {
        std::cerr << "frame bus: NvBufSurfaceMap failed" << std::endl;
        return false;
    }
    if (NvBufSurfaceSyncForCpu(surf, 0, -1) != 0) {
        std::cerr << "frame bus: NvBufSurfaceSyncForCpu failed" << std::endl;
        NvBufSurfaceUnMap(surf, 0, -1);
        return false;
    }

    const NvBufSurfaceParams& params = surf->surfaceList[0];
    const NvBufSurfacePlaneParams& planes = params.planeParams;
    const uint32_t y_pitch = planes.pitch[0];
    const uint32_t uv_pitch = planes.pitch[1] > 0 ? planes.pitch[1] : y_pitch;
    uint8_t* y = static_cast<uint8_t*>(params.mappedAddr.addr[0]);
    uint8_t* uv = static_cast<uint8_t*>(params.mappedAddr.addr[1]);
    if (!uv && y) {
        uv = y + static_cast<size_t>(y_pitch) * height;
    }
    if (!y || !uv || y_pitch < width || uv_pitch < width) {
        std::cerr << "frame bus: mapped NV12 surface has invalid planes" << std::endl;
        NvBufSurfaceUnMap(surf, 0, -1);
        return false;
    }

    const std::string tmp_path = path + ".tmp";
    std::ofstream out(tmp_path.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "frame bus: failed to open " << tmp_path << std::endl;
        NvBufSurfaceUnMap(surf, 0, -1);
        return false;
    }
    for (uint32_t row = 0; row < height; ++row) {
        out.write(reinterpret_cast<const char*>(y + static_cast<size_t>(row) * y_pitch), width);
    }
    for (uint32_t row = 0; row < height / 2; ++row) {
        out.write(reinterpret_cast<const char*>(uv + static_cast<size_t>(row) * uv_pitch), width);
    }
    out.flush();
    const bool write_ok = static_cast<bool>(out);
    out.close();
    NvBufSurfaceUnMap(surf, 0, -1);
    if (!write_ok) {
        std::cerr << "frame bus: failed to write " << tmp_path << std::endl;
        return false;
    }
    if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
        std::cerr << "frame bus: failed to rename " << tmp_path
                  << " to " << path << ": " << std::strerror(errno) << std::endl;
        return false;
    }
    return true;
}

bool should_publish_frame_bus(
    const std::string& frame_bus_dir,
    uint32_t frame_bus_every_n,
    uint64_t logical_index
) {
    return (
        !frame_bus_dir.empty()
        && frame_bus_every_n > 0
        && (logical_index % frame_bus_every_n) == 0
    );
}

bool publish_frame_bus_cluster(
    const std::string& frame_bus_dir,
    uint32_t frame_bus_every_n,
    const std::vector<std::unique_ptr<FrameBundle>>& cluster,
    uint64_t logical_index,
    uint64_t min_sof,
    uint64_t max_sof,
    bool recording,
    int episode_idx
) {
    if (!should_publish_frame_bus(frame_bus_dir, frame_bus_every_n, logical_index)) {
        return true;
    }
    if (cluster.empty()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_frame_bus_mutex);
    if (!mkdir_p(frame_bus_dir)) {
        return false;
    }

    const uint64_t publish_seq = g_frame_bus_success_count;
    const uint64_t slot = publish_seq % 2;
    std::vector<std::pair<std::string, FrameBundle*>> published;
    published.reserve(cluster.size());
    for (const auto& item : cluster) {
        FrameBundle* bundle = item.get();
        if (!bundle || !bundle->cam || !bundle->dma_buffer) {
            return false;
        }
        std::ostringstream raw_name;
        raw_name << "slot" << slot << "_" << bundle->cam->name << ".nv12";
        const std::string raw_path = path_join(frame_bus_dir, raw_name.str());
        if (!copy_nv12_to_file(
                bundle->dma_buffer,
                bundle->cam->width,
                bundle->cam->height,
                raw_path)) {
            return false;
        }
        published.push_back(std::make_pair(raw_path, bundle));
    }

    const CamCtx* first_cam = cluster.front()->cam;
    std::ostringstream json;
    json << "{\n"
         << "  \"version\": 1,\n"
         << "  \"publish_seq\": " << publish_seq << ",\n"
         << "  \"slot\": " << slot << ",\n"
         << "  \"recording\": " << (recording ? "true" : "false") << ",\n"
         << "  \"episode_index\": " << episode_idx << ",\n"
         << "  \"logical_frame_index\": " << logical_index << ",\n"
         << "  \"sync_source\": \"sof_tsc_ns\",\n"
         << "  \"format\": \"nv12\",\n"
         << "  \"width\": " << (first_cam ? first_cam->width : 0) << ",\n"
         << "  \"height\": " << (first_cam ? first_cam->height : 0) << ",\n"
         << "  \"min_sof_tsc_ns\": " << min_sof << ",\n"
         << "  \"max_sof_tsc_ns\": " << max_sof << ",\n"
         << "  \"max_delta_ns\": " << (max_sof >= min_sof ? max_sof - min_sof : 0) << ",\n"
         << "  \"cameras\": {\n";
    for (size_t i = 0; i < published.size(); ++i) {
        FrameBundle* bundle = published[i].second;
        json << "    \"" << json_escape(bundle->cam->name) << "\": {"
             << "\"path\": \"" << json_escape(published[i].first) << "\", "
             << "\"camera\": \"" << json_escape(bundle->cam->name) << "\", "
             << "\"logical_frame_index\": " << logical_index << ", "
             << "\"local_frame_number\": " << bundle->meta.local_frame_number << ", "
             << "\"sensor_timestamp_ns\": " << bundle->meta.sensor_timestamp_ns << ", "
             << "\"sof_tsc_ns\": " << bundle->meta.sof_tsc_ns << ", "
             << "\"eof_tsc_ns\": " << bundle->meta.eof_tsc_ns << ", "
             << "\"internal_frame_count\": " << bundle->meta.internal_frame_count
             << "}" << (i + 1 == published.size() ? "\n" : ",\n");
    }
    json << "  }\n"
         << "}\n";

    const std::string latest_path = path_join(frame_bus_dir, "latest_cluster.json");
    if (!write_text_atomic(latest_path, json.str())) {
        return false;
    }
    g_frame_bus_success_count += 1;
    return true;
}

bool publish_frame_bus_cluster(
    const Options& options,
    const std::vector<std::unique_ptr<FrameBundle>>& cluster,
    uint64_t logical_index,
    uint64_t min_sof,
    uint64_t max_sof,
    bool recording,
    int episode_idx
) {
    return publish_frame_bus_cluster(
        options.frame_bus_dir,
        options.frame_bus_every_n,
        cluster,
        logical_index,
        min_sof,
        max_sof,
        recording,
        episode_idx
    );
}

bool publish_preview_frame_bus_cluster(
    const Options& options,
    const std::vector<std::unique_ptr<FrameBundle>>& cluster,
    uint64_t logical_index,
    uint64_t min_sof,
    uint64_t max_sof
) {
    return publish_frame_bus_cluster(
        options.preview_frame_bus_dir,
        options.preview_frame_bus_every_n,
        cluster,
        logical_index,
        min_sof,
        max_sof,
        false,
        -1
    );
}

using FrameQueue = std::deque<std::unique_ptr<FrameBundle>>;
using FrameQueues = std::vector<FrameQueue>;


void append_unmatched_drop_detail(
    std::string* detail,
    const CamCtx* cam,
    uint64_t sof,
    uint64_t local_min,
    uint64_t local_max,
    uint64_t tolerance_ns
) {
    if (!detail || detail->size() > 512) {
        return;
    }
    if (!detail->empty()) {
        *detail += "; ";
    }
    uint64_t delta_to_max = local_max >= sof ? local_max - sof : sof - local_max;
    std::ostringstream oss;
    oss << "drop " << (cam ? cam->name : "unknown")
        << " sof=" << sof
        << " range=[" << local_min << "," << local_max << "]"
        << " delta_to_max_ns=" << delta_to_max
        << " tolerance_ns=" << tolerance_ns;
    *detail += oss.str();
}

bool acquire_cluster(
    const std::vector<std::unique_ptr<CamCtx>>& cameras,
    FrameQueues* queues,
    const Options& options,
    std::vector<std::unique_ptr<FrameBundle>>* out,
    uint64_t* min_sof,
    uint64_t* max_sof,
    uint64_t* dropped_unmatched,
    std::string* failure_detail
) {
    out->clear();
    *min_sof = std::numeric_limits<uint64_t>::max();
    *max_sof = 0;
    const uint64_t timeout_ns = static_cast<uint64_t>(options.frame_timeout_ms) * 1000ULL * 1000ULL;
    if (queues->size() != cameras.size()) {
        const std::string detail = "internal queue/camera count mismatch";
        if (failure_detail) {
            *failure_detail = detail;
        }
        std::cerr << detail << std::endl;
        return false;
    }

    const size_t max_iterations = std::max<size_t>(240, cameras.size() * 240);
    for (size_t iteration = 0; iteration < max_iterations && !g_stop_requested.load(); ++iteration) {
        for (size_t i = 0; i < cameras.size(); ++i) {
            if ((*queues)[i].empty()) {
                std::unique_ptr<FrameBundle> bundle = acquire_one(
                    cameras[i].get(), timeout_ns, failure_detail
                );
                if (!bundle) {
                    return false;
                }
                (*queues)[i].push_back(std::move(bundle));
            }
        }

        uint64_t local_min = std::numeric_limits<uint64_t>::max();
        uint64_t local_max = 0;
        size_t local_min_idx = 0;
        for (size_t i = 0; i < queues->size(); ++i) {
            uint64_t sof = (*queues)[i].front()->meta.sof_tsc_ns;
            if (sof < local_min) {
                local_min = sof;
                local_min_idx = i;
            }
            local_max = std::max(local_max, sof);
        }
        if (local_min == std::numeric_limits<uint64_t>::max() || local_max < local_min) {
            if (failure_detail) {
                *failure_detail = "invalid SOF range while forming full cluster";
            }
            return false;
        }

        if ((local_max - local_min) <= options.tolerance_ns) {
            *min_sof = local_min;
            *max_sof = local_max;
            for (size_t i = 0; i < queues->size(); ++i) {
                out->push_back(std::move((*queues)[i].front()));
                (*queues)[i].pop_front();
            }
            return true;
        }

        const uint64_t threshold = local_max > options.tolerance_ns ? local_max - options.tolerance_ns : 0;
        bool dropped = false;
        for (size_t i = 0; i < queues->size(); ++i) {
            if (!(*queues)[i].empty() && (*queues)[i].front()->meta.sof_tsc_ns < threshold) {
                append_unmatched_drop_detail(
                    failure_detail,
                    cameras[i].get(),
                    (*queues)[i].front()->meta.sof_tsc_ns,
                    local_min,
                    local_max,
                    options.tolerance_ns
                );
                release_bundle((*queues)[i].front().get());
                (*queues)[i].pop_front();
                if (dropped_unmatched) {
                    *dropped_unmatched += 1;
                }
                dropped = true;
            }
        }
        if (!dropped) {
            append_unmatched_drop_detail(
                failure_detail,
                cameras[local_min_idx].get(),
                (*queues)[local_min_idx].front()->meta.sof_tsc_ns,
                local_min,
                local_max,
                options.tolerance_ns
            );
            release_bundle((*queues)[local_min_idx].front().get());
            (*queues)[local_min_idx].pop_front();
            if (dropped_unmatched) {
                *dropped_unmatched += 1;
            }
        }
    }

    if (g_stop_requested.load()) {
        return false;
    }
    {
        std::ostringstream oss;
        oss << "failed to form a full SOF cluster within iteration budget; "
            << "iterations=" << max_iterations
            << ", tolerance_ns=" << options.tolerance_ns;
        if (failure_detail) {
            *failure_detail = oss.str();
        }
        std::cerr << oss.str() << std::endl;
    }
    return false;
}

bool push_cluster(
    std::vector<std::unique_ptr<FrameBundle>>& cluster,
    uint64_t logical_index,
    uint64_t min_sof,
    uint64_t max_sof
) {
    const uint64_t center = min_sof / 2 + max_sof / 2 + ((min_sof & 1) && (max_sof & 1) ? 1 : 0);
    for (auto& bundle : cluster) {
        if (!bundle->cam->encoder->push_buffer(bundle->buffer, bundle->dma_buffer, logical_index)) {
            bundle->cam->ok = false;
            release_bundle(bundle.get());
            return false;
        }
        bundle->buffer = nullptr;
        bundle->dma_buffer = nullptr;
    }
    for (auto& bundle : cluster) {
        uint64_t sof = bundle->meta.sof_tsc_ns;
        uint64_t delta = sof > center ? sof - center : center - sof;
        bundle->cam->max_abs_delta_ns = std::max(bundle->cam->max_abs_delta_ns, delta);
        bundle->cam->accepted_frames += 1;
        bundle->cam->csv << bundle->cam->name << ","
                         << logical_index << ","
                         << bundle->meta.local_frame_number << ","
                         << bundle->meta.sensor_timestamp_ns << ","
                         << bundle->meta.sof_tsc_ns << ","
                         << bundle->meta.eof_tsc_ns << ","
                         << bundle->meta.internal_frame_count << "\n";
    }
    return true;
}

void write_manifest(
    const Options& options,
    const std::vector<std::unique_ptr<CamCtx>>& cameras,
    bool ok,
    const std::string& failure,
    uint64_t target_frames,
    uint64_t actual_frames,
    uint64_t dropped_before_start,
    uint64_t dropped_after_stop
) {
    std::string path = options.episode_dir + "/online_sync_manifest.json";
    std::ofstream out(path.c_str(), std::ios::out | std::ios::trunc);
    if (!out) {
        std::cerr << "failed to write " << path << std::endl;
        return;
    }
    out << "{\n"
        << "  \"ok\": " << (ok ? "true" : "false") << ",\n"
        << "  \"failure\": \"" << json_escape(failure) << "\",\n"
        << "  \"fps\": " << options.fps << ",\n"
        << "  \"target_frames\": " << target_frames << ",\n"
        << "  \"actual_frames\": " << actual_frames << ",\n"
        << "  \"sync_source\": \"sof_tsc_ns\",\n"
        << "  \"tolerance_ns\": " << options.tolerance_ns << ",\n"
        << "  \"frame_timeout_ms\": " << options.frame_timeout_ms << ",\n"
        << "  \"missing_frame_policy\": \"" << json_escape(options.missing_frame_policy) << "\",\n"
        << "  \"stop_mode\": \"" << json_escape(options.stop_mode) << "\",\n"
        << "  \"frame_bus_enabled\": " << (!options.frame_bus_dir.empty() ? "true" : "false") << ",\n"
        << "  \"frame_bus_dir\": \"" << json_escape(options.frame_bus_dir) << "\",\n"
        << "  \"frame_bus_every_n\": " << options.frame_bus_every_n << ",\n"
        << "  \"dropped_clusters_before_start\": " << dropped_before_start << ",\n"
        << "  \"dropped_clusters_after_stop\": " << dropped_after_stop << ",\n";
    out << "  \"active_cameras\": [";
    for (size_t i = 0; i < cameras.size(); ++i) {
        out << (i ? ", " : "") << "\"" << json_escape(cameras[i]->name) << "\"";
    }
    out << "],\n";
    out << "  \"frame_count_by_camera\": {\n";
    for (size_t i = 0; i < cameras.size(); ++i) {
        out << "    \"" << json_escape(cameras[i]->name) << "\": " << cameras[i]->accepted_frames
            << (i + 1 == cameras.size() ? "\n" : ",\n");
    }
    out << "  },\n";
    out << "  \"max_abs_delta_ns_by_camera\": {\n";
    for (size_t i = 0; i < cameras.size(); ++i) {
        out << "    \"" << json_escape(cameras[i]->name) << "\": " << cameras[i]->max_abs_delta_ns
            << (i + 1 == cameras.size() ? "\n" : ",\n");
    }
    out << "  }\n";
    out << "}\n";
}

void release_all_queues(FrameQueues* queues) {
    if (!queues) {
        return;
    }
    for (auto& queue : *queues) {
        while (!queue.empty()) {
            release_bundle(queue.front().get());
            queue.pop_front();
        }
    }
}

bool open_episode_outputs_for_all(
    const std::vector<std::unique_ptr<CamCtx>>& cameras,
    const Options& options,
    std::string* failure
) {
    if (!mkdir_p(options.episode_dir)) {
        if (failure) {
            *failure = "failed to create episode directory";
        }
        return false;
    }
    for (auto& cam : cameras) {
        ISensorMode* i_sensor_mode = interface_cast<ISensorMode>(cam->sensor_mode);
        if (!i_sensor_mode) {
            if (failure) {
                *failure = cam->name + ": sensor mode unavailable while opening episode outputs";
            }
            return false;
        }
        if (!open_episode_outputs(cam.get(), options, i_sensor_mode->getResolution(), options.episode_dir)) {
            if (failure) {
                *failure = cam->name + ": failed to open episode outputs";
            }
            for (auto& opened : cameras) {
                close_episode_outputs(opened.get());
            }
            return false;
        }
    }
    return true;
}

bool start_encoders_for_all(
    const std::vector<std::unique_ptr<CamCtx>>& cameras,
    std::string* failure
) {
    for (auto& cam : cameras) {
        if (!cam->encoder || !cam->encoder->start()) {
            if (failure) {
                *failure = cam->name + ": encoder start failed";
            }
            return false;
        }
    }
    return true;
}

bool stop_encoders_for_all(
    const std::vector<std::unique_ptr<CamCtx>>& cameras,
    std::string* failure
) {
    bool ok = true;
    for (auto& cam : cameras) {
        if (cam->csv) {
            cam->csv.flush();
        }
    }
    for (auto& cam : cameras) {
        if (cam->encoder && !cam->encoder->stop()) {
            ok = false;
            if (failure && failure->empty()) {
                *failure = "encoder EOS failed";
            }
        }
        if (cam->csv) {
            cam->csv.flush();
            cam->csv.close();
        }
        if (cam->encoder) {
            cam->encoder.reset();
        }
    }
    return ok;
}

struct RecordingResult {
    bool ok = true;
    std::string failure;
    uint64_t target_frames = 0;
    uint64_t actual_frames = 0;
    uint64_t dropped_before_start = 0;
    uint64_t dropped_after_stop = 0;
    bool stopped_by_request = false;
};

RecordingResult record_episode(
    const std::vector<std::unique_ptr<CamCtx>>& cameras,
    const Options& options,
    FrameQueues* queues,
    uint32_t startup_full_clusters,
    std::atomic<bool>* episode_stop_requested,
    int episode_idx
) {
    RecordingResult result;
    result.target_frames = options.frames;
    const uint64_t target_frames = options.frames;
    std::vector<std::unique_ptr<FrameBundle>> cluster;
    uint64_t logical_index = 0;
    uint64_t warmup_full_clusters = 0;
    uint64_t min_sof = 0;
    uint64_t max_sof = 0;

    if (!open_episode_outputs_for_all(cameras, options, &result.failure)) {
        result.ok = false;
        write_manifest(options, cameras, false, result.failure, target_frames, 0, 0, 0);
        return result;
    }
    if (!start_encoders_for_all(cameras, &result.failure)) {
        result.ok = false;
        stop_encoders_for_all(cameras, &result.failure);
        write_manifest(options, cameras, false, result.failure, target_frames, 0, 0, 0);
        return result;
    }

    auto startup_deadline = (
        std::chrono::steady_clock::now()
        + std::chrono::milliseconds(options.startup_timeout_ms)
    );
    while (
        warmup_full_clusters < startup_full_clusters
        && !g_stop_requested.load()
        && !(episode_stop_requested && episode_stop_requested->load())
    ) {
        uint64_t unmatched_drops = 0;
        std::string cluster_failure;
        bool got = acquire_cluster(
            cameras, queues, options, &cluster, &min_sof, &max_sof,
            &unmatched_drops, &cluster_failure
        );
        result.dropped_before_start += unmatched_drops;
        if (got) {
            warmup_full_clusters += 1;
            result.dropped_before_start += 1;
            release_cluster(&cluster);
            continue;
        }
        if (g_stop_requested.load() || (episode_stop_requested && episode_stop_requested->load())) {
            result.stopped_by_request = true;
            break;
        }
        if (std::chrono::steady_clock::now() >= startup_deadline) {
            result.ok = false;
            result.failure = "startup did not produce enough full SOF clusters";
            if (!cluster_failure.empty()) {
                result.failure += ": " + cluster_failure;
            }
            break;
        }
    }

    if (result.ok && !result.stopped_by_request && !g_stop_requested.load()) {
        if (episode_idx >= 0) {
            std::cerr << "recording started idx=" << episode_idx << std::endl;
        } else {
            std::cerr << "recording started" << std::endl;
        }
    }

    while (
        result.ok
        && !g_stop_requested.load()
        && !(episode_stop_requested && episode_stop_requested->load())
        && (target_frames == 0 || logical_index < target_frames)
    ) {
        uint64_t unmatched_drops = 0;
        std::string cluster_failure;
        if (!acquire_cluster(
                cameras, queues, options, &cluster, &min_sof, &max_sof,
                &unmatched_drops, &cluster_failure
            )) {
            if (g_stop_requested.load() && target_frames == 0) {
                result.stopped_by_request = true;
                break;
            }
            result.ok = false;
            result.failure = "missing or out-of-tolerance full SOF cluster after recording start";
            if (!cluster_failure.empty()) {
                result.failure += ": " + cluster_failure;
            }
            break;
        }
        if (unmatched_drops > 0) {
            result.ok = false;
            std::ostringstream oss;
            oss << "missing full SOF cluster inside recording window; unmatched_drops="
                << unmatched_drops;
            if (!cluster_failure.empty()) {
                oss << ": " << cluster_failure;
            }
            result.failure = oss.str();
            release_cluster(&cluster);
            break;
        }
        if (!publish_frame_bus_cluster(
                options,
                cluster,
                logical_index,
                min_sof,
                max_sof,
                true,
                episode_idx)) {
            std::cerr << "frame bus: failed to publish recording cluster "
                      << logical_index << std::endl;
        }
        if (!push_cluster(cluster, logical_index, min_sof, max_sof)) {
            result.ok = false;
            result.failure = "failed to push synchronized cluster into encoder";
            release_cluster(&cluster);
            break;
        }
        logical_index += 1;
    }

    if ((episode_stop_requested && episode_stop_requested->load()) || (g_stop_requested.load() && target_frames == 0)) {
        result.stopped_by_request = true;
    }

    result.actual_frames = logical_index;
    bool stop_ok = stop_encoders_for_all(cameras, &result.failure);
    if (!stop_ok) {
        result.ok = false;
    }

    if (target_frames > 0 && logical_index != target_frames && !result.stopped_by_request) {
        result.ok = false;
        if (result.failure.empty()) {
            result.failure = "actual frame count did not reach target";
        }
    }
    for (const auto& cam : cameras) {
        if (cam->accepted_frames != logical_index) {
            result.ok = false;
            if (result.failure.empty()) {
                result.failure = "camera frame counts diverged";
            }
        }
    }

    write_manifest(
        options,
        cameras,
        result.ok,
        result.failure,
        target_frames,
        logical_index,
        result.dropped_before_start,
        result.dropped_after_stop
    );
    return result;
}

bool start_argus_repeat(const std::vector<std::unique_ptr<CamCtx>>& cameras) {
    for (auto& cam : cameras) {
        if (cam->i_session->repeat(cam->request.get()) != STATUS_OK) {
            std::cerr << cam->name << ": repeat request failed" << std::endl;
            return false;
        }
    }
    return true;
}

void stop_argus_repeat(const std::vector<std::unique_ptr<CamCtx>>& cameras) {
    for (auto& cam : cameras) {
        if (cam->i_session) {
            cam->i_session->stopRepeat();
            if (cam->i_buffer_stream) {
                cam->i_buffer_stream->endOfStream();
            }
        }
    }
}

void push_command(const Command& command) {
    std::lock_guard<std::mutex> lock(g_command_mutex);
    g_commands.push_back(command);
}

bool pop_command(Command* command) {
    std::lock_guard<std::mutex> lock(g_command_mutex);
    if (g_commands.empty()) {
        return false;
    }
    *command = g_commands.front();
    g_commands.pop_front();
    return true;
}

void stdin_command_loop() {
    std::string line;
    while (std::getline(std::cin, line)) {
        std::stringstream ss(line);
        std::string verb;
        ss >> verb;
        if (verb.empty()) {
            continue;
        }
        if (verb == "START") {
            Command command;
            command.type = CommandType::Start;
            ss >> command.idx >> command.frames >> command.episode_dir;
            if (command.episode_dir.empty()) {
                std::cerr << "invalid START command: " << line << std::endl;
                continue;
            }
            g_episode_stop_requested.store(false);
            g_preview_requested.store(false);
            push_command(command);
        } else if (verb == "STOP") {
            g_episode_stop_requested.store(true);
        } else if (verb == "PREVIEW_ON") {
            g_preview_requested.store(true);
            push_command(Command{CommandType::PreviewOn, 0, 0, ""});
        } else if (verb == "PREVIEW_OFF") {
            g_preview_requested.store(false);
            push_command(Command{CommandType::PreviewOff, 0, 0, ""});
        } else if (verb == "QUIT") {
            g_quit_requested.store(true);
            g_stop_requested.store(true);
            g_preview_requested.store(false);
            push_command(Command{CommandType::Quit, 0, 0, ""});
            break;
        } else {
            std::cerr << "unknown persistent command: " << line << std::endl;
        }
    }
    if (std::cin.eof()) {
        g_quit_requested.store(true);
        g_stop_requested.store(true);
    }
}

int run_finite(
    const std::vector<std::unique_ptr<CamCtx>>& cameras,
    const Options& options
) {
    if (!start_argus_repeat(cameras)) {
        return 4;
    }
    FrameQueues queues(cameras.size());
    RecordingResult result = record_episode(
        cameras, options, &queues, options.startup_full_clusters, nullptr, -1
    );
    release_all_queues(&queues);
    stop_argus_repeat(cameras);
    if (!result.ok) {
        std::cerr << "recording failed: " << result.failure << std::endl;
        std::cerr.flush();
        std::cout.flush();
        _exit(5);
    }
    std::cerr << "online sync frames captured per camera: " << result.actual_frames << std::endl;
    std::cerr.flush();
    std::cout.flush();
    _exit(0);
}

int run_persistent(
    const std::vector<std::unique_ptr<CamCtx>>& cameras,
    const Options& options
) {
    if (!start_argus_repeat(cameras)) {
        return 4;
    }
    FrameQueues queues(cameras.size());
    std::thread(stdin_command_loop).detach();
    std::cerr << "persistent ready" << std::endl;
    uint64_t idle_logical_index = 0;

    while (!g_quit_requested.load() && !g_stop_requested.load()) {
        Command command;
        if (pop_command(&command)) {
            if (command.type == CommandType::Quit) {
                break;
            }
            if (command.type == CommandType::PreviewOn) {
                std::cerr << "preview publishing enabled" << std::endl;
                continue;
            }
            if (command.type == CommandType::PreviewOff) {
                std::cerr << "preview publishing disabled" << std::endl;
                continue;
            }
            if (command.type == CommandType::Start) {
                Options episode_options = options;
                episode_options.frames = command.frames;
                episode_options.episode_dir = command.episode_dir;
                g_episode_stop_requested.store(false);
                RecordingResult result = record_episode(
                    cameras, episode_options, &queues,
                    episode_options.startup_full_clusters,
                    &g_episode_stop_requested,
                    static_cast<int>(command.idx)
                );
                release_all_queues(&queues);
                std::cerr << "episode done idx=" << command.idx
                          << " ok=" << (result.ok ? "true" : "false")
                          << " frames=" << result.actual_frames;
                if (!result.failure.empty()) {
                    std::cerr << " failure=" << result.failure;
                }
                std::cerr << std::endl;
                continue;
            }
        }

        std::vector<std::unique_ptr<FrameBundle>> idle_cluster;
        uint64_t min_sof = 0;
        uint64_t max_sof = 0;
        uint64_t unmatched_drops = 0;
        std::string failure;
        if (!acquire_cluster(
                cameras, &queues, options, &idle_cluster, &min_sof, &max_sof,
                &unmatched_drops, &failure
            )) {
            if (g_quit_requested.load() || g_stop_requested.load()) {
                break;
            }
            std::cerr << "persistent idle cluster miss: " << failure << std::endl;
            release_cluster(&idle_cluster);
            release_all_queues(&queues);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        if (!publish_frame_bus_cluster(
                options,
                idle_cluster,
                idle_logical_index,
                min_sof,
                max_sof,
                false,
                -1)) {
            std::cerr << "frame bus: failed to publish idle cluster "
                      << idle_logical_index << std::endl;
        }
        if (
            g_preview_requested.load()
            && !publish_preview_frame_bus_cluster(
                options,
                idle_cluster,
                idle_logical_index,
                min_sof,
                max_sof
            )
        ) {
            std::cerr << "preview frame bus: failed to publish idle cluster "
                      << idle_logical_index << std::endl;
        }
        idle_logical_index += 1;
        release_cluster(&idle_cluster);
    }

    release_all_queues(&queues);
    stop_argus_repeat(cameras);
    std::cerr << "persistent exiting" << std::endl;
    std::cerr.flush();
    std::cout.flush();
    _exit(0);
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
    g_egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (g_egl_display == EGL_NO_DISPLAY) {
        std::cerr << "Cannot get EGL display" << std::endl;
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

    if (options.persistent) {
        return run_persistent(cameras, options);
    }
    return run_finite(cameras, options);
}
