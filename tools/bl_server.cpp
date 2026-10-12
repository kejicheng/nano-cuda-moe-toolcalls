// An OpenAI-compatible HTTP server (one model, one request at a time):
//   GET  /v1/models
//   POST /v1/chat/completions   messages, stream, temperature, top_p, top_k, max_tokens / max_completion_tokens, seed,
//                               reasoning_effort, chat_template_kwargs.enable_thinking; non-standard: spec (MTP drafts)
//   GET  /health
// Thinking is off unless the request asks (enable_thinking: true or a reasoning_effort); the thinking comes back as
// reasoning_content. The engine keeps the conversation between requests: a request that continues it (the same
// messages plus new ones) only feeds the new part.
//   bl-server --model SHARD1.gguf [--host 127.0.0.1] [--port 8080] [--ctx 32768] [--name NAME] [--web DIR|none]
//             [--expert-cache FILE]
// Switching models while running: GET /bl/models lists the quantizations next to --model (files "...-00001-of-N.gguf"
// with all their shards) and the state; POST /bl/models {"id": "IQ3_S"} answers at once and swaps the model in the
// background once the current reply is done (chat requests meanwhile get 503); if the new one fails to load, the old
// one comes back. The chat page has it in its settings panel.
// The chat page (web/index.html) is served at /. With --expert-cache, the expert cache (which experts sit in VRAM) is
// saved after every reply and restored at the next start. Off by default: on new text it did not raise the hit rate
// (88.4% restored vs 89.0% from the profile, 2026-10-05).
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "bl/chat.h"
#include "httplib.h"
#include "nlohmann/json.hpp"

using json = nlohmann::ordered_json;

namespace {

std::string content_text(const json & c) {   // a message's content: a string, or parts of which the text ones count
    if (c.is_string()) return c.get<std::string>();
    std::string s;
    if (c.is_array())
        for (const auto & part : c)
            if (part.is_object() && part.value("type", "") == "text") s += part.value("text", "");
    return s;
}

// a model file's id: the quantization in "...GSQ-RCO-<id>-00001-of-N.gguf", else the name before "-00001-of-"
std::string model_id(const std::string & path) {
    const std::string f = path.substr(path.rfind('/') + 1);
    const size_t end = f.find("-00001-of-"), at = f.rfind("GSQ-RCO-", end);
    if (end == std::string::npos) return f;
    return at == std::string::npos ? f.substr(0, end) : f.substr(at + 8, end - at - 8);
}

struct ModelFile { std::string id, path; double gb; };
// the models in shard 1's directory whose shards are all there (sizes through symlinks)
std::vector<ModelFile> list_models(const std::string & shard1) {
    const size_t sl = shard1.rfind('/');
    const std::string dir = sl == std::string::npos ? "." : shard1.substr(0, sl);
    std::vector<ModelFile> r;
    DIR * d = opendir(dir.c_str());
    if (!d) return r;
    while (const dirent * e = readdir(d)) {
        const std::string f = e->d_name;
        const size_t at = f.find("-00001-of-");
        if (at == std::string::npos || f.size() < 5 || f.compare(f.size() - 5, 5, ".gguf") != 0) continue;
        const int n = std::atoi(f.c_str() + at + 10);
        bool all = n >= 1;
        double gb = 0;
        for (int k = 1; all && k <= n; ++k) {
            char part[16];
            std::snprintf(part, sizeof part, "-%05d-of-", k);
            std::string g = f;
            g.replace(at, 10, part);
            struct stat st {};
            all = ::stat((dir + "/" + g).c_str(), &st) == 0;
            gb += st.st_size / 1e9;
        }
        if (all) r.push_back({model_id(f), dir + "/" + f, gb});
    }
    closedir(d);
    std::sort(r.begin(), r.end(), [](const ModelFile & a, const ModelFile & b) { return a.id < b.id; });
    return r;
}

json error_json(const std::string & msg, const std::string & type = "invalid_request_error") {
    return json{{"error", {{"message", msg}, {"type", type}}}};
}

std::string trim(const std::string & s) {
    const auto b = s.find_first_not_of(" \t\r\n"), e = s.find_last_not_of(" \t\r\n");
    return b == std::string::npos ? "" : s.substr(b, e - b + 1);
}

// The system-turn preamble the model's template adds when a request carries tools. The tool objects go into
// <tools>...</tools> verbatim (each its OpenAI object), followed by the call-format instructions the model was
// trained on: <tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>.
std::string tools_block(const json & tools, const json & choice) {
    std::string s = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
    for (const auto & t : tools) s += "\n" + t.dump();
    s += "\n</tools>";
    s += "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
         "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
         "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
         "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
         "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested "
         "within <tool_call></tool_call> XML tags\n"
         "- Required parameters MUST be specified\n"
         "- You may provide optional reasoning for your function call in natural language BEFORE the function call, "
         "but NOT after\n"
         "- If there is no function call available, answer the question like normal with your current knowledge and "
         "do not tell the user about function calls\n</IMPORTANT>";
    if (choice == "required") s += "\n\nYou MUST call one of the functions above.";
    else if (choice.is_object() && choice.contains("function") && choice["function"].is_object())
        s += "\n\nYou MUST call the function `" + choice["function"].value("name", std::string()) + "`.";
    return s;
}

// The model's reply, split into its text and its calls. The call blocks are the template's XML form (with a JSON
// body accepted as a fallback); each call is an OpenAI tool_call object, its arguments a JSON object string.
// Anything the model wrote before the first call stays in content_out; the blocks themselves are removed from it.
json parse_tool_calls(const std::string & text, std::string & content_out, long & id_seq) {
    json calls = json::array();
    std::string content;
    const std::string open = "<tool_call>", close = "</tool_call>";
    size_t pos = 0;
    while (true) {
        const size_t b = text.find(open, pos);
        if (b == std::string::npos) { content += text.substr(pos); break; }
        content += text.substr(pos, b - pos);
        const size_t e = text.find(close, b + open.size());
        if (e == std::string::npos) { content += text.substr(b); break; }   // unterminated: leave it as text
        const std::string block = text.substr(b + open.size(), e - b - open.size());
        std::string fname;
        json args = json::object();
        const size_t fs = block.find("<function=");
        if (fs != std::string::npos) {
            const size_t fe = block.find('>', fs);
            fname = block.substr(fs + 10, fe == std::string::npos ? std::string::npos : fe - fs - 10);
            size_t p = fe == std::string::npos ? block.size() : fe + 1;
            while (true) {   // <parameter=KEY>\nVALUE\n</parameter>
                const size_t ps = block.find("<parameter=", p);
                if (ps == std::string::npos) break;
                const size_t pe = block.find('>', ps);
                if (pe == std::string::npos) break;
                const std::string key = block.substr(ps + 11, pe - ps - 11);
                size_t vs = pe + 1;
                if (vs < block.size() && block[vs] == '\n') ++vs;
                const size_t ve = block.find("</parameter>", vs);
                std::string raw = block.substr(vs, ve == std::string::npos ? std::string::npos : ve - vs);
                if (!raw.empty() && raw.back() == '\n') raw.pop_back();
                json v;
                try { v = json::parse(raw); } catch (...) { v = raw; }   // a number/object, else the text as is
                args[key] = v;
                p = ve == std::string::npos ? block.size() : ve + 12;
            }
        } else {   // a JSON body: {"name": "...", "arguments": {...}}
            try {
                const json j = json::parse(trim(block));
                fname = j.value("name", std::string());
                if (j.contains("arguments")) {
                    const json & a = j["arguments"];
                    args = a.is_string() ? json::parse(a.get<std::string>()) : a;
                }
            } catch (...) {}
        }
        if (!fname.empty())
            calls.push_back({{"id", "call_" + std::to_string(++id_seq)}, {"type", "function"},
                             {"function", {{"name", fname}, {"arguments", args.dump()}}}});
        else
            content += text.substr(b, e - b + close.size());   // not a call after all: keep it as text
        pos = e + close.size();
    }
    content_out = trim(content);
    return calls;
}

// an OpenAI tool_calls array as the template wants it in an assistant turn (for feeding a history back in)
std::string calls_to_xml(const json & calls) {
    std::string out;
    for (const auto & tc : calls) {
        const json f = tc.contains("function") ? tc["function"] : tc;
        const std::string name = f.value("name", std::string());
        if (name.empty()) continue;
        json args;
        try { const json & a = f.value("arguments", json("{}")); args = a.is_string() ? json::parse(a.get<std::string>()) : a; }
        catch (...) { args = json::object(); }
        out += (out.empty() ? std::string() : std::string("\n")) + "<tool_call>\n<function=" + name + ">\n";
        for (const auto & [k, v] : args.items()) {
            out += "<parameter=" + k + ">\n" + (v.is_string() ? v.get<std::string>() : v.dump()) + "\n</parameter>\n";
        }
        out += "</function>\n</tool_call>";
    }
    return out;
}

}  // namespace

int main(int argc, char ** argv) {
    std::string model, host = "127.0.0.1", name, web, cache_file;   // name: from the model's id unless given
    int port = 8080, ctx = 32768;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--model") model = next();
        else if (a == "--host") host = next();
        else if (a == "--port") port = std::stoi(next());
        else if (a == "--ctx") ctx = std::stoi(next());
        else if (a == "--name") name = next();
        else if (a == "--web") web = next();   // the chat page's directory (default: ../web next to the executable)
        else if (a == "--expert-cache") cache_file = next();
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (model.empty()) { std::fprintf(stderr, "usage: %s --model SHARD1.gguf [--host H] [--port P] [--ctx N]\n", argv[0]); return 2; }

    // the loaded model; replaced (under `busy`) by a switch
    struct Loaded { std::unique_ptr<bl::Text> text; std::unique_ptr<bl::Engine> eng; std::unique_ptr<bl::Chat> chat; std::string path; };
    Loaded cur;
    const bool fixed_name = !name.empty();
    std::mutex state_mu;   // guards the switch state below and the model's name (read without waiting for `busy`)
    std::string current, target, state = "ready", switch_error;
    std::atomic<bool> switching{false};
    auto model_name = [&] { std::lock_guard<std::mutex> lk(state_mu); return name; };
    auto load = [&](const std::string & path) {   // the old one goes first: two do not fit
        cur.chat.reset();
        cur.eng.reset();
        cur.text.reset();
        std::fprintf(stderr, "loading %s ...\n", path.c_str());
        cur.text = std::make_unique<bl::Text>(path);
        cur.eng = std::make_unique<bl::Engine>(path, ctx, cache_file);
        cur.chat = std::make_unique<bl::Chat>(*cur.eng, *cur.text, ctx);
        cur.path = path;
        std::lock_guard<std::mutex> lk(state_mu);
        current = model_id(path);
        if (!fixed_name) {
            name = "qwen3.8-flash-" + current;
            std::transform(name.begin(), name.end(), name.begin(), [](unsigned char ch) { return std::tolower(ch); });
        }
    };
    load(model);
    auto save_cache = [&] {   // after a reply (under `busy`): the next start begins with this cache
        if (cache_file.empty() || !cur.eng) return;
        try { cur.eng->save_cache(cache_file); } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); }
    };
    std::mutex busy;   // one generation (or a model switch) at a time
    std::atomic<long> seq{0};
    long tool_seq = 0;   // tool_call ids (only under `busy`)

    httplib::Server srv;
    if (web.empty()) {   // ../web relative to this executable (build/bl-server -> web/)
        char exe[4096];
        const ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
        if (n > 0) {
            std::string d(exe, static_cast<size_t>(n));
            d = d.substr(0, d.rfind('/'));
            web = d.substr(0, d.rfind('/')) + "/web";
        }
    }
    if (web != "none" && srv.set_mount_point("/", web)) std::fprintf(stderr, "chat page: %s\n", web.c_str());
    srv.Get("/health", [](const httplib::Request &, httplib::Response & res) { res.set_content("{\"status\":\"ok\"}", "application/json"); });
    srv.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content(json{{"object", "list"}, {"data", json::array({{{"id", model_name()}, {"object", "model"}, {"owned_by", "boundless"}}})}}.dump(),
                        "application/json");
    });
    srv.Get("/bl/models", [&](const httplib::Request &, httplib::Response & res) {
        json list = json::array();
        for (const auto & m : list_models(model)) list.push_back({{"id", m.id}, {"gb", std::round(m.gb * 10) / 10}});
        std::lock_guard<std::mutex> lk(state_mu);
        res.set_content(json{{"current", current}, {"state", state}, {"target", target}, {"error", switch_error}, {"models", list}}.dump(),
                        "application/json");
    });
    srv.Post("/bl/models", [&](const httplib::Request & req, httplib::Response & res) {
        std::string id;
        try { id = json::parse(req.body).at("id").get<std::string>(); }
        catch (...) { res.status = 400; res.set_content(error_json("expected {\"id\": \"<quantization>\"}").dump(), "application/json"); return; }
        std::string path;
        for (const auto & m : list_models(model)) if (m.id == id) path = m.path;
        if (path.empty()) { res.status = 404; res.set_content(error_json("no model " + id).dump(), "application/json"); return; }
        {
            std::lock_guard<std::mutex> lk(state_mu);
            if (switching) { res.status = 409; res.set_content(error_json("already switching to " + target).dump(), "application/json"); return; }
            if (id == current && state == "ready") { res.set_content(json{{"state", "ready"}, {"current", current}}.dump(), "application/json"); return; }
            switching = true;
            state = "loading";
            target = id;
            switch_error.clear();
        }
        std::thread([&, path, id] {
            std::lock_guard<std::mutex> lk(busy);   // after the current reply
            const auto t0 = std::chrono::steady_clock::now();
            const std::string prev = cur.path;
            std::string err;
            std::fprintf(stderr, "switching to %s\n", id.c_str());
            try { load(path); } catch (const std::exception & e) { err = e.what(); }
            if (!err.empty()) {   // the old one back
                std::fprintf(stderr, "switch to %s failed: %s; reloading %s\n", id.c_str(), err.c_str(), prev.c_str());
                try { load(prev); } catch (const std::exception & e) { err += std::string("; reloading the previous model failed too: ") + e.what(); }
            } else {
                std::fprintf(stderr, "switched to %s in %.1f s\n", id.c_str(),
                             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            }
            std::lock_guard<std::mutex> sl(state_mu);
            state = cur.chat ? "ready" : "failed";
            switch_error = err;
            switching = false;
        }).detach();
        res.status = 202;
        res.set_content(json{{"state", "loading"}, {"target", id}}.dump(), "application/json");
    });
    srv.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & res) {
        if (switching) {
            res.status = 503;
            std::lock_guard<std::mutex> lk(state_mu);
            res.set_content(error_json("switching the model to " + target + ", try again in a minute", "server_busy").dump(), "application/json");
            return;
        }
        const std::string name = model_name();
        json body;
        try { body = json::parse(req.body); } catch (...) { res.status = 400; res.set_content(error_json("invalid JSON").dump(), "application/json"); return; }
        std::vector<bl::Text::Message> msgs;
        bl::ChatParams cp;
        try {
            for (const auto & m : body.at("messages")) {
                bl::Text::Message tm{m.at("role").get<std::string>(), content_text(m.value("content", json(""))),
                                     m.value("reasoning_content", std::string()), std::string()};
                if (m.contains("tool_calls") && m["tool_calls"].is_array())
                    tm.tool_calls = calls_to_xml(m["tool_calls"]);   // an assistant turn's calls, back into the template
                msgs.push_back(std::move(tm));
            }
            cp.sampling.temperature = body.value("temperature", 0.7f);
            cp.sampling.top_p = body.value("top_p", 0.8f);
            cp.sampling.top_k = body.value("top_k", 20);
            cp.sampling.seed = body.value("seed", 0ull);
            cp.max_new = body.value("max_completion_tokens", body.value("max_tokens", 4096));
            cp.spec = body.value("spec", 3);
            const bool think = body.contains("chat_template_kwargs") && body["chat_template_kwargs"].value("enable_thinking", false);
            std::string effort = body.value("reasoning_effort", std::string());
            if (effort == "high") effort = "xhigh";
            cp.think = !effort.empty() ? effort : think ? "xhigh" : "";
            if (body.contains("tools") && body["tools"].is_array() && !body["tools"].empty()) {
                const json choice = body.value("tool_choice", json("auto"));
                if (!(choice.is_string() && choice.get<std::string>() == "none"))   // "none" (or no tools): text only
                    cp.tools = tools_block(body["tools"], choice);
            }
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(error_json(e.what()).dump(), "application/json");
            return;
        }
        const bool stream = body.value("stream", false);
        const std::string id = "chatcmpl-" + std::to_string(++seq);
        const long created = static_cast<long>(std::time(nullptr));
        auto usage = [](const bl::ChatResult & r) {
            return json{{"prompt_tokens", r.n_prompt + r.n_reused}, {"completion_tokens", r.n_generated},
                        {"total_tokens", r.n_prompt + r.n_reused + r.n_generated},
                        {"prompt_tokens_details", {{"cached_tokens", r.n_reused}}}};
        };
        if (!stream) {
            std::lock_guard<std::mutex> lk(busy);
            try {
                if (!cur.chat) throw std::runtime_error("no model loaded (the last switch failed)");
                const bl::ChatResult r = cur.chat->reply(msgs, cp);
                std::string content;
                const json calls = parse_tool_calls(r.content, content, tool_seq);
                json msg{{"role", "assistant"}, {"content", content}};
                if (!r.reasoning.empty()) msg["reasoning_content"] = r.reasoning;
                if (!calls.empty()) msg["tool_calls"] = calls;
                const std::string finish = !calls.empty() ? "tool_calls" : r.stopped_eog ? "stop" : "length";
                res.set_content(json{{"id", id}, {"object", "chat.completion"}, {"created", created}, {"model", name},
                                     {"choices", json::array({{{"index", 0}, {"message", msg}, {"finish_reason", finish}}})},
                                     {"usage", usage(r)}}.dump(), "application/json");
                std::fprintf(stderr, "%s: %d prompt (+%d reused) in %.2f s, %d tokens at %.1f tok/s\n", id.c_str(), r.n_prompt, r.n_reused,
                             r.prompt_s, r.n_generated, r.gen_s > 0 ? (r.n_generated - 1) / r.gen_s : 0.0);
                save_cache();
            } catch (const std::exception & e) {
                res.status = 400;
                res.set_content(error_json(e.what()).dump(), "application/json");
            }
            return;
        }
        res.set_chunked_content_provider("text/event-stream", [&, msgs, cp, id, created, name](size_t, httplib::DataSink & sink) {
            std::lock_guard<std::mutex> lk(busy);
            auto send = [&](const json & j) {
                const std::string s = "data: " + j.dump() + "\n\n";
                return sink.write(s.data(), s.size());
            };
            auto chunk = [&](const json & delta, const json & finish) {
                return json{{"id", id}, {"object", "chat.completion.chunk"}, {"created", created}, {"model", name},
                            {"choices", json::array({{{"index", 0}, {"delta", delta}, {"finish_reason", finish}}})}};
            };
            json calls = json::array();
            std::string pend;   // content held back: it may be the beginning of a call block
            // The model's calls come after any text, in <tool_call>...</tool_call> blocks. Send the text in front of
            // one as soon as it is complete, turn each complete block into a tool_calls delta, and hold back a tail
            // that could still become the marker (so it never leaks into content).
            auto flush = [&]() -> bool {
                while (true) {
                    const size_t b = pend.find("<tool_call>");
                    if (b == std::string::npos) break;
                    const size_t e = pend.find("</tool_call>", b);
                    if (e == std::string::npos) {   // incomplete: send the text before it, keep the rest
                        if (b > 0 && !send(chunk(json{{"content", pend.substr(0, b)}}, nullptr))) return false;
                        pend.erase(0, b);
                        return true;
                    }
                    if (b > 0 && !send(chunk(json{{"content", pend.substr(0, b)}}, nullptr))) return false;
                    const std::string block = pend.substr(b, e + 12 - b);
                    pend.erase(0, e + 12);
                    std::string discard;
                    for (auto & c : parse_tool_calls(block, discard, tool_seq)) {
                        c["index"] = static_cast<int>(calls.size());
                        calls.push_back(c);
                        if (!send(chunk(json{{"tool_calls", json::array({c})}}, nullptr))) return false;
                    }
                }
                size_t hold = 0;
                for (size_t k = std::min(pend.size(), static_cast<size_t>(11)); k > 0; --k)
                    if (pend.compare(pend.size() - k, k, "<tool_call>", k) == 0) { hold = k; break; }
                if (pend.size() > hold && !send(chunk(json{{"content", pend.substr(0, pend.size() - hold)}}, nullptr))) return false;
                pend.erase(0, pend.size() - hold);
                return true;
            };
            send(chunk({{"role", "assistant"}, {"content", ""}}, nullptr));
            try {
                if (!cur.chat) throw std::runtime_error("no model loaded (the last switch failed)");
                const bl::ChatResult r = cur.chat->reply(msgs, cp, [&](const std::string & s, bool reasoning) {
                    if (reasoning) return send(chunk(json{{"reasoning_content", s}}, nullptr));
                    pend += s;
                    return flush();   // false: the client left
                });
                std::string tail;
                const json rest = parse_tool_calls(pend, tail, tool_seq);
                if (!tail.empty() && !send(chunk(json{{"content", tail}}, nullptr))) throw std::runtime_error("closed");
                for (auto c : rest) {
                    c["index"] = static_cast<int>(calls.size());
                    calls.push_back(c);
                    if (!send(chunk(json{{"tool_calls", json::array({c})}}, nullptr))) throw std::runtime_error("closed");
                }
                json last = chunk(json::object(), !calls.empty() ? json("tool_calls") : json(r.stopped_eog ? "stop" : "length"));
                last["usage"] = usage(r);
                send(last);
                std::fprintf(stderr, "%s (stream): %d prompt (+%d reused) in %.2f s, %d tokens at %.1f tok/s\n", id.c_str(), r.n_prompt,
                             r.n_reused, r.prompt_s, r.n_generated, r.gen_s > 0 ? (r.n_generated - 1) / r.gen_s : 0.0);
                save_cache();
            } catch (const std::exception & e) {
                send(error_json(e.what()));
            }
            const std::string done = "data: [DONE]\n\n";
            sink.write(done.data(), done.size());
            sink.done();
            return true;
        });
    });
    std::fprintf(stderr, "listening on http://%s:%d (OpenAI API: /v1/chat/completions)\n", host.c_str(), port);
    if (!srv.listen(host, port)) { std::fprintf(stderr, "cannot listen on %s:%d\n", host.c_str(), port); return 1; }
    return 0;
}
