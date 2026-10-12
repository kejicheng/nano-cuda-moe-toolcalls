// Telemetry behind the Monitor: a background thread samples the hardware once a second and keeps the last 60
// readings of every series, for the page's sparklines. The idea, the series names and the "nothing here can stop
// the server" rule follow Strata's serve/telemetry.py.
// The GPU is read through NVIDIA's own NVML library (libnvidia-ml.so.1, installed with every driver) opened at
// runtime with dlopen - what Strata does with ctypes - so nothing is linked against the driver and no package is
// needed. CPU, RAM and disk come from /proc. Anything unreadable stays NaN, and the JSON turns that into null.
#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace bl {

// the sampled series, in the order the Monitor lists them
struct HardwareReading {
    bool        gpu_ok = false;
    std::string gpu_name;
    double gpu_util = NaN(), gpu_mem_used = NaN(), gpu_mem_total = NaN(), gpu_temp = NaN();
    double gpu_power = NaN(), gpu_power_limit = NaN();
    double gpu_pcie_gen = NaN(), gpu_pcie_gen_max = NaN(), gpu_pcie_width = NaN();
    double gpu_pcie_rx_mb = NaN(), gpu_pcie_tx_mb = NaN();
    double cpu = NaN(), ram_used = NaN(), ram_total = NaN();
    double disk_read_mb = NaN(), disk_write_mb = NaN();
    // the server's own series, filled by the extra callback before every sample
    double tok_s = NaN(), tok_s_mean = NaN(), prefill_tok_s = NaN();

    static constexpr double NaN() { return std::numeric_limits<double>::quiet_NaN(); }
};

// what does not change while the server runs
struct HardwareStatic {
    std::string gpu_name, cpu_name, gpu_note;
    int  cores = 0, threads = 0;
    bool nvml = false;   // the driver's NVML library was found and initialized
};

class Telemetry {
public:
    static constexpr int kHistory = 60;   // a minute of readings at 1 Hz

    // extra: the server's own series, recorded each second (called on the sampler thread, must not throw)
    explicit Telemetry(std::function<void(HardwareReading &)> extra = nullptr);
    ~Telemetry();
    Telemetry(const Telemetry &) = delete;
    Telemetry & operator=(const Telemetry &) = delete;

    HardwareReading now() const;                                  // the last reading
    std::map<std::string, std::vector<double>> history() const;   // per series, oldest first
    const HardwareStatic & statics() const { return static_; }

    // the reading, once, on the calling thread (for a caller that wants NVML without the thread)
    static HardwareReading read_now(int gpu_index = 0);

private:
    void loop();
    void sample(HardwareReading & r);
    void push(const char * key, double v);

    std::function<void(HardwareReading &)> extra_;
    HardwareStatic static_;
    mutable std::mutex mu_;
    HardwareReading now_;
    std::map<std::string, std::deque<double>> hist_;
    std::atomic<bool> stop_{false};
    std::thread th_;
    long long prev_cpu_idle_ = -1, prev_cpu_total_ = -1;   // /proc/stat deltas
    long long prev_disk_read_ = -1, prev_disk_write_ = -1;
    double    prev_disk_t_ = 0;
};

}  // namespace bl
