/*
 * Libargus per-frame metadata capture tool for Thor GMSL2 cameras.
 *
 * This tool opens one Argus CaptureSession per camera and writes one CSV sidecar
 * per camera:
 *
 *   cam_06.argus_frame_metadata.csv
 *
 * The sidecar schema matches tools/thor/gmsl2/argus_frame_sync.py.  It is the
 * metadata half of the production synchronized recorder: the final recorder
 * must associate these rows with encoded video frames from the same Libargus
 * capture owner.
 *
 * Build on Thor:
 *
 *   g++ -std=c++14 -O2 \
 *     -I/usr/src/jetson_multimedia_api/argus/include \
 *     -I/usr/src/jetson_multimedia_api/argus/samples/utils \
 *     tools/thor/gmsl2/argus_frame_metadata_capture.cpp \
 *     /usr/src/jetson_multimedia_api/argus/samples/utils/ArgusHelpers.cpp \
 *     -L/usr/lib/aarch64-linux-gnu/tegra -lnvargus_socketclient -lpthread \
 *     -o /tmp/argus_frame_metadata_capture
 *
 * Example:
 *
 *   /tmp/argus_frame_metadata_capture --sids 6,7 --frames 120 --out-dir /tmp/argus_meta
 */

#include "ArgusHelpers.h"

#include <Argus/Argus.h>
#include <Argus/Ext/InternalFrameCount.h>
#include <Argus/Ext/SensorTimestampTsc.h>
#include <EGLStream/EGLStream.h>

#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>

using namespace Argus;
using namespace EGLStream;
using namespace ArgusSamples;

namespace {

struct Options {
    std::vector<uint32_t> sids;
    uint32_t frames = 120;
    uint32_t sensor_mode = 0;
    std::string out_dir = ".";
};

struct FrameMetadata {
    uint64_t encoded_frame_index = 0;
    uint64_t local_frame_number = 0;
    uint64_t sensor_timestamp_ns = 0;
    uint64_t sof_tsc_ns = 0;
    uint64_t eof_tsc_ns = 0;
    uint64_t internal_frame_count = 0;
};

struct CamCtx {
    uint32_t sid = 0;
    std::string camera_name;
    CameraDevice* camera_device = nullptr;
    SensorMode* sensor_mode = nullptr;
    UniqueObj<CaptureSession> session;
    ICaptureSession* i_session = nullptr;
    UniqueObj<OutputStreamSettings> stream_settings;
    UniqueObj<OutputStream> stream;
    UniqueObj<FrameConsumer> consumer;
    IFrameConsumer* i_consumer = nullptr;
    UniqueObj<Request> request;
    std::ofstream csv;
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

std::string camera_name(uint32_t sid) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "cam_%02u", sid);
    return std::string(buf);
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
        } else if (arg == "--out-dir") {
            const char* value = require_value("--out-dir");
            if (!value) return false;
            options->out_dir = value;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "usage: " << argv[0]
                      << " --sids 6,7 [--frames 120] [--sensor-mode 0] [--out-dir DIR]"
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
    if (options->frames == 0) {
        std::cerr << "--frames must be > 0" << std::endl;
        return false;
    }
    return true;
}

bool ensure_out_dir(const std::string& out_dir) {
    struct stat st;
    if (stat(out_dir.c_str(), &st) == 0) {
        if (S_ISDIR(st.st_mode)) {
            return true;
        }
        std::cerr << "--out-dir exists but is not a directory: " << out_dir << std::endl;
        return false;
    }
    if (mkdir(out_dir.c_str(), 0775) == 0) {
        return true;
    }
    std::cerr << "failed to create --out-dir: " << out_dir << std::endl;
    return false;
}

bool init_camera(
    ICameraProvider* provider,
    UniqueObj<CameraProvider>& provider_obj,
    CamCtx* cam,
    uint32_t sensor_mode_index,
    const std::string& out_dir
) {
    cam->camera_name = camera_name(cam->sid);
    cam->camera_device = ArgusHelpers::getCameraDevice(provider_obj.get(), cam->sid);
    if (!cam->camera_device) {
        std::cerr << cam->camera_name << ": camera device unavailable" << std::endl;
        return false;
    }
    cam->sensor_mode = ArgusHelpers::getSensorMode(cam->camera_device, sensor_mode_index);
    if (!cam->sensor_mode) {
        std::cerr << cam->camera_name << ": sensor mode unavailable" << std::endl;
        return false;
    }
    ISensorMode* i_sensor_mode = interface_cast<ISensorMode>(cam->sensor_mode);
    if (!i_sensor_mode) {
        std::cerr << cam->camera_name << ": ISensorMode unavailable" << std::endl;
        return false;
    }

    cam->session = UniqueObj<CaptureSession>(provider->createCaptureSession(cam->camera_device));
    cam->i_session = interface_cast<ICaptureSession>(cam->session);
    if (!cam->i_session) {
        std::cerr << cam->camera_name << ": createCaptureSession failed" << std::endl;
        return false;
    }

    cam->stream_settings = UniqueObj<OutputStreamSettings>(
        cam->i_session->createOutputStreamSettings(STREAM_TYPE_EGL)
    );
    IEGLOutputStreamSettings* i_stream_settings =
        interface_cast<IEGLOutputStreamSettings>(cam->stream_settings);
    if (!i_stream_settings) {
        std::cerr << cam->camera_name << ": createOutputStreamSettings failed" << std::endl;
        return false;
    }
    i_stream_settings->setPixelFormat(PIXEL_FMT_YCbCr_420_888);
    i_stream_settings->setResolution(i_sensor_mode->getResolution());
    i_stream_settings->setMode(EGL_STREAM_MODE_FIFO);
    i_stream_settings->setMetadataEnable(true);

    cam->stream = UniqueObj<OutputStream>(
        cam->i_session->createOutputStream(cam->stream_settings.get())
    );
    if (!cam->stream) {
        std::cerr << cam->camera_name << ": createOutputStream failed" << std::endl;
        return false;
    }

    cam->consumer = UniqueObj<FrameConsumer>(FrameConsumer::create(cam->stream.get()));
    cam->i_consumer = interface_cast<IFrameConsumer>(cam->consumer);
    if (!cam->i_consumer) {
        std::cerr << cam->camera_name << ": FrameConsumer failed" << std::endl;
        return false;
    }

    cam->request = UniqueObj<Request>(cam->i_session->createRequest());
    IRequest* i_request = interface_cast<IRequest>(cam->request);
    if (!i_request) {
        std::cerr << cam->camera_name << ": createRequest failed" << std::endl;
        return false;
    }
    i_request->enableOutputStream(cam->stream.get());
    ISourceSettings* i_source_settings = interface_cast<ISourceSettings>(cam->request);
    if (!i_source_settings) {
        std::cerr << cam->camera_name << ": ISourceSettings unavailable" << std::endl;
        return false;
    }
    i_source_settings->setSensorMode(cam->sensor_mode);

    std::string csv_path = out_dir + "/" + cam->camera_name + ".argus_frame_metadata.csv";
    cam->csv.open(csv_path.c_str(), std::ios::out | std::ios::trunc);
    if (!cam->csv) {
        std::cerr << cam->camera_name << ": cannot open " << csv_path << std::endl;
        return false;
    }
    cam->csv << "camera,encoded_frame_index,local_frame_number,sensor_timestamp_ns,"
             << "sof_tsc_ns,eof_tsc_ns,internal_frame_count\n";
    return true;
}

bool start_camera(CamCtx* cam) {
    if (!cam->i_session) return false;
    if (cam->i_session->repeat(cam->request.get()) != STATUS_OK) {
        std::cerr << cam->camera_name << ": repeat request failed" << std::endl;
        return false;
    }
    return true;
}

bool acquire_metadata(CamCtx* cam, uint64_t encoded_frame_index, FrameMetadata* out) {
    UniqueObj<Frame> frame(cam->i_consumer->acquireFrame());
    if (!frame) {
        std::cerr << cam->camera_name << ": acquireFrame returned null" << std::endl;
        return false;
    }

    IFrame* i_frame = interface_cast<IFrame>(frame);
    IArgusCaptureMetadata* i_argus_meta = interface_cast<IArgusCaptureMetadata>(frame);
    if (!i_frame || !i_argus_meta) {
        std::cerr << cam->camera_name << ": frame metadata interface missing" << std::endl;
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

void write_metadata(CamCtx* cam, const FrameMetadata& meta) {
    cam->csv << cam->camera_name << ","
             << meta.encoded_frame_index << ","
             << meta.local_frame_number << ","
             << meta.sensor_timestamp_ns << ","
             << meta.sof_tsc_ns << ","
             << meta.eof_tsc_ns << ","
             << meta.internal_frame_count << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_args(argc, argv, &options)) {
        return 2;
    }
    if (!ensure_out_dir(options.out_dir)) {
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
        if (!init_camera(provider, provider_obj, cam.get(), options.sensor_mode, options.out_dir)) {
            return 3;
        }
        cameras.push_back(std::move(cam));
    }

    for (auto& cam : cameras) {
        if (!start_camera(cam.get())) {
            return 4;
        }
    }

    for (uint64_t frame_index = 0; frame_index < options.frames; ++frame_index) {
        for (auto& cam : cameras) {
            FrameMetadata meta;
            if (!acquire_metadata(cam.get(), frame_index, &meta)) {
                return 5;
            }
            write_metadata(cam.get(), meta);
        }
    }

    for (auto& cam : cameras) {
        if (cam->i_session) {
            cam->i_session->stopRepeat();
            cam->i_session->waitForIdle();
        }
        if (cam->csv) {
            cam->csv.flush();
            cam->csv.close();
        }
    }
    cameras.clear();
    provider_obj.reset();
    std::cerr << "frames captured per camera: " << options.frames << std::endl;
    return 0;
}
