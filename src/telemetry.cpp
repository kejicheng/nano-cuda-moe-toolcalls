// The Monitor's sampler (see bl/telemetry.h). Nothing in here may throw into the server: every reading is
// best-effort and a failure only leaves that series NaN.
#include "bl/telemetry.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <chrono>
#include <string>
#include <unistd.h>

namespace bl {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// ------------------------------------------------------------------------------------------------ NVML (dlopen)
// The driver's own library, opened the way Strata opens it with ctypes: no pip package, no link-time dependency,
// and a machine without an NVIDIA driver simply reports no GPU readings.
struct Nvml {
    struct Util { unsigned int gpu, memory; };
    struct Mem  { unsigned long long total, free, used; };

    void * lib = nullptr;
    void * dev = nullptr;

    int (*init)() = nullptr;
    int (*get_handle)(unsigned int, void **) = nullptr;
    int (*get_name)(void *, char *, unsigned int) = nullptr;
    int (*get_util)(void *, Util *) = nullptr;
    int (*get_mem)(void *, Mem *) = nullptr;
    int (*get_temp)(void *, int, unsigned int *) = nullptr;
    int (*get_power)(void *, unsigned int *) = nullptr;
    int (*get_power_limit)(void *, unsigned int *) = nullptr;
    int (*get_link_gen)(void *, unsigned int *) = nullptr;
    int (*get_link_gen_max)(void *, unsigned int *) = nullptr;
    int (*get_link_width)(void *, unsigned int *) = nullptr;
    int (*get_pcie_throughput)(void *, int, unsigned int *) = nullptr;

    explicit Nvml(int index) {
        lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
        if (!lib) lib = dlopen("libnvidia-ml.so", RTLD_LAZY);
        if (!lib) return;
        *(void **) &init = dlsym(lib, "nvmlInit_v2");
        if (!init) *(void **) &init = dlsym(lib, "nvmlInit");
        *(void **) &get_handle = dlsym(lib, "nvmlDeviceGetHandleByIndex_v2");
        if (!get_handle) *(void **) &get_handle = dlsym(lib, "nvmlDeviceGetHandleByIndex");
        if (!init || !get_handle || init() != 0) { close(); return; }
        *(void **) &get_name = dlsym(lib, "nvmlDeviceGetName");
        *(void **) &get_util = dlsym(lib, "nvmlDeviceGetUtilizationRates");
        *(void **) &get_mem = dlsym(lib, "nvmlDeviceGetMemoryInfo");
        *(void **) &get_temp = dlsym(lib, "nvmlDeviceGetTemperature");
        *(void **) &get_power = dlsym(lib, "nvmlDeviceGetPowerUsage");
        *(void **) &get_power_limit = dlsym(lib, "nvmlDeviceGetEnforcedPowerLimit");
        *(void **) &get_link_gen = dlsym(lib, "nvmlDeviceGetCurrPcieLinkGeneration");
        *(void **) &get_link_gen_max = dlsym(lib, "nvmlDeviceGetMaxPcieLinkGeneration");
        *(void **) &get_link_width = dlsym(lib, "nvmlDeviceGetCurrPcieLinkWidth");
        *(void **) &get_pcie_throughput = dlsym(lib, "nvmlDeviceGetPcieThroughput");
        void * h = nullptr;
        if (get_handle(static_cast<unsigned int>(index), &h) != 0 || !h) { close(); return; }
        dev = h;
    }

    ~Nvml() { close(); }

    void close() {
        if (lib) dlclose(lib);   // NVML's own cleanup is left to the driver (nvmlShutdown is not needed here)
        lib = nullptr;
        dev = nullptr;
    }

    bool ok() const { return lib && dev; }

    std::string name() const {
        if (!get_name) return {};
        char buf[96] = {};
        if (get_name(dev, buf, sizeof buf) != 0) return {};
        return std::string(buf);
    }

    double num(int (*fn)(void *, unsigned int *)) const {
        unsigned int v = 0;
        return (fn && fn(dev, &v) == 0) ? static_cast<double>(v) : kNaN;
    }

    void fill(HardwareReading & r) const {
        Util u{};
        if (get_util && get_util(dev, &u) == 0) r.gpu_util = u.gpu;
        Mem m{};
        if (get_mem && get_mem(dev, &m) == 0) { r.gpu_mem_used = static_cast<double>(m.used); r.gpu_mem_total = static_cast<double>(m.total); }
        if (get_temp) {
            unsigned int mv = 0;
            if (get_temp(dev, 0, &mv) == 0) r.gpu_temp = mv;   // NVML_TEMPERATURE_GPU
        }
        const double mw = num(get_power);
        if (!std::isnan(mw)) r.gpu_power = mw / 1000.0;
        const double lim = num(get_power_limit);
        if (!std::isnan(lim)) r.gpu_power_limit = lim / 1000.0;
        r.gpu_pcie_gen = num(get_link_gen);
        r.gpu_pcie_gen_max = num(get_link_gen_max);
        r.gpu_pcie_width = num(get_link_width);
        unsigned int rx = 0, tx = 0;
        if (get_pcie_throughput) {
            if (get_pcie_throughput(dev, 1, &rx) == 0) r.gpu_pcie_rx_mb = rx / 1024.0;   // NVML_PCIE_UTIL_RX_BYTES, KB/s
            if (get_pcie_throughput(dev, 0, &tx) == 0) r.gpu_pcie_tx_mb = tx / 1024.0;
        }
    }
};

// ------------------------------------------------------------------------------------------------ /proc
std::string slurp(const char * path) {
    std::string out;
    std::FILE * f = std::fopen(path, "rb");
    if (!f) return out;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
    std::fclose(f);
    return out;
}

std::string first_cpu_name() {
    const std::string s = slurp("/proc/cpuinfo");
    const size_t at = s.find("model name");
    if (at == std::string::npos) return {};
    const size_t c = s.find(':', at);
    const size_t e = s.find('\n', c);
    if (c == std::string::npos) return {};
    std::string v = s.substr(c + 1, e - c - 1);
    while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) v.erase(v.begin());
    while (!v.empty() && (v.back() == ' ' || v.back() == '\r')) v.pop_back();
    return v;
}

// physical cores: the (physical id, core id) pairs cpuinfo lists, one per logical processor
int physical_cores() {
    const std::string s = slurp("/proc/cpuinfo");
    if (s.empty()) return 0;
    std::vector<std::pair<std::string, std::string>> seen;
    std::string phys, core;
    size_t at = 0;
    while (at < s.size()) {
        const size_t e = s.find('\n', at);
        const std::string line = s.substr(at, e == std::string::npos ? std::string::npos : e - at);
        if (line.rfind("physical id", 0) == 0) phys = line.substr(line.find(':') + 1);
        else if (line.rfind("core id", 0) == 0) core = line.substr(line.find(':') + 1);
        else if (line.empty()) {
            if (!phys.empty() || !core.empty()) {
                const auto k = std::make_pair(phys, core);
                bool dup = false;
                for (const auto & p : seen) dup = dup || p == k;
                if (!dup) seen.push_back(k);
            }
            phys.clear();
            core.clear();
        }
        if (e == std::string::npos) break;
        at = e + 1;
    }
    return static_cast<int>(seen.size());
}

// CPU: idle+iowait over all, from /proc/stat's first line; the rate needs the previous call
struct CpuTimes { long long idle = -1, total = -1; };
CpuTimes cpu_times() {
    CpuTimes t;
    const std::string s = slurp("/proc/stat");
    if (s.rfind("cpu ", 0) != 0) return t;
    long long v[10] = {};
    int n = 0;
    const char * p = s.c_str() + 4;
    while (n < 10 && p && *p && *p != '\n') {
        char * end = nullptr;
        const long long x = std::strtoll(p, &end, 10);
        if (end == p) break;
        v[n++] = x;
        p = end;
    }
    if (n < 5) return t;
    long long total = 0;
    for (int i = 0; i < n; ++i) total += v[i];
    t.idle = v[3] + v[4];   // idle + iowait
    t.total = total;
    return t;
}

struct Ram { double used = kNaN, total = kNaN; };
Ram ram() {
    Ram r;
    const std::string s = slurp("/proc/meminfo");
    double total = -1, avail = -1;
    size_t at = 0;
    while (at < s.size()) {
        const size_t e = s.find('\n', at);
        const std::string line = s.substr(at, e == std::string::npos ? std::string::npos : e - at);
        if (line.rfind("MemTotal:", 0) == 0) total = std::strtod(line.c_str() + 9, nullptr) * 1024;
        else if (line.rfind("MemAvailable:", 0) == 0) avail = std::strtod(line.c_str() + 13, nullptr) * 1024;
        if (e == std::string::npos) break;
        at = e + 1;
    }
    if (total > 0 && avail >= 0) { r.total = total; r.used = total - avail; }
    return r;
}

// Disk: the sectors every whole device has read and written (/proc/diskstats); partitions and the loop/ram/dm
// pseudo devices are skipped so nothing is counted twice
bool whole_device(const std::string & n) {
    if (n.rfind("loop", 0) == 0 || n.rfind("ram", 0) == 0 || n.rfind("dm-", 0) == 0 || n.rfind("zram", 0) == 0) return false;
    if (n.rfind("sd", 0) == 0 || n.rfind("hd", 0) == 0 || n.rfind("vd", 0) == 0 || n.rfind("xvd", 0) == 0)
        return n.find_first_of("0123456789") == std::string::npos;
    if (n.rfind("nvme", 0) == 0 || n.rfind("mmcblk", 0) == 0) return n.find('p') == std::string::npos;
    return false;
}

void disk_bytes(long long & read, long long & written) {
    read = written = 0;
    const std::string s = slurp("/proc/diskstats");
    size_t at = 0;
    while (at < s.size()) {
        const size_t e = s.find('\n', at);
        const std::string line = s.substr(at, e == std::string::npos ? std::string::npos : e - at);
        long long maj = 0, min = 0, rc = 0, rm = 0, sr = 0, msr = 0, wc = 0, wm = 0, sw = 0;
        char name[64] = {};
        if (std::sscanf(line.c_str(), "%lld %lld %63s %lld %lld %lld %lld %lld %lld %lld",
                        &maj, &min, name, &rc, &rm, &sr, &msr, &wc, &wm, &sw) == 10 && whole_device(name)) {
            read += sr * 512;      // sectors read
            written += sw * 512;   // sectors written
        }
        if (e == std::string::npos) break;
        at = e + 1;
    }
}

}  // namespace

HardwareReading Telemetry::read_now(int gpu_index) {
    HardwareReading r;
    Nvml nv(gpu_index);
    if (nv.ok()) {
        r.gpu_ok = true;
        r.gpu_name = nv.name();
        nv.fill(r);
    }
    return r;
}

Telemetry::Telemetry(std::function<void(HardwareReading &)> extra) : extra_(std::move(extra)) {
    static_.gpu_name = "?";
    static_.cpu_name = first_cpu_name();
    static_.threads = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
    static_.cores = physical_cores();
    if (static_.cores <= 0) static_.cores = static_.threads;
    {
        Nvml nv(0);
        if (nv.ok()) {
            static_.nvml = true;
            static_.gpu_name = nv.name();
            static_.gpu_name = static_.gpu_name.empty() ? "NVIDIA GPU" : static_.gpu_name;
        } else {
            static_.gpu_name.clear();
            static_.gpu_note = "no NVML: libnvidia-ml.so.1 was not found (no NVIDIA driver, or a container without "
                               "the driver's libraries mounted) - GPU readings are absent";
        }
    }
    HardwareReading first;
    sample(first);
    {
        std::lock_guard<std::mutex> lk(mu_);
        now_ = first;
    }
    th_ = std::thread([this] { loop(); });
}

Telemetry::~Telemetry() { stop_.store(true); if (th_.joinable()) th_.join(); }

void Telemetry::push(const char * key, double v) {
    auto & d = hist_[key];
    if (d.size() >= static_cast<size_t>(kHistory)) d.pop_front();
    d.push_back(v);
}

void Telemetry::sample(HardwareReading & r) {
    static int gpu_index = -1;
    if (gpu_index < 0) {
        const char * e = std::getenv("BL_GPU_INDEX");
        gpu_index = e ? std::atoi(e) : 0;
    }
    r = HardwareReading();
    Nvml nv(gpu_index);
    if (nv.ok()) {
        r.gpu_ok = true;
        r.gpu_name = nv.name();
        nv.fill(r);
    }
    const CpuTimes t = cpu_times();
    if (t.total > 0 && prev_cpu_total_ > 0 && t.total > prev_cpu_total_) {
        const double busy = 1.0 - static_cast<double>(t.idle - prev_cpu_idle_) / static_cast<double>(t.total - prev_cpu_total_);
        r.cpu = std::max(0.0, std::min(100.0, busy * 100.0));
    }
    if (t.total > 0) { prev_cpu_idle_ = t.idle; prev_cpu_total_ = t.total; }
    const Ram m = ram();
    r.ram_used = m.used;
    r.ram_total = m.total;
    long long rd = 0, wr = 0;
    disk_bytes(rd, wr);
    const double secs = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                                std::chrono::steady_clock::now().time_since_epoch()).count()) / 1000.0;
    if (prev_disk_read_ >= 0 && secs > prev_disk_t_) {
        const double dt = secs - prev_disk_t_;
        r.disk_read_mb = static_cast<double>(rd - prev_disk_read_) / dt / 1048576.0;
        r.disk_write_mb = static_cast<double>(wr - prev_disk_write_) / dt / 1048576.0;
    }
    prev_disk_read_ = rd;
    prev_disk_write_ = wr;
    prev_disk_t_ = secs;
    if (extra_) {
        try { extra_(r); } catch (...) {}   // telemetry must never take the server down
    }
}

void Telemetry::loop() {
    while (!stop_.load()) {
        HardwareReading r;
        sample(r);
        std::lock_guard<std::mutex> lk(mu_);
        now_ = r;
        push("gpu_util", r.gpu_util);
        push("gpu_mem_used", r.gpu_mem_used);
        push("gpu_temp", r.gpu_temp);
        push("gpu_power", r.gpu_power);
        push("gpu_pcie_rx_mb", r.gpu_pcie_rx_mb);
        push("cpu", r.cpu);
        push("ram_used", r.ram_used);
        push("disk_read_mb", r.disk_read_mb);
        push("tok_s", r.tok_s);
        push("prefill_tok_s", r.prefill_tok_s);
        for (int i = 0; i < 10 && !stop_.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

HardwareReading Telemetry::now() const {
    std::lock_guard<std::mutex> lk(mu_);
    return now_;
}

std::map<std::string, std::vector<double>> Telemetry::history() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::map<std::string, std::vector<double>> out;
    for (const auto & [k, d] : hist_) out[k] = std::vector<double>(d.begin(), d.end());
    return out;
}

}  // namespace bl
