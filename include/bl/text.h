// Text in and out: the model's tokenizer (llama.cpp's, loaded vocab-only from the GGUF: no weights), the chat
// template (the model's Jinja template for text messages, written out), and UTF-8-safe streaming of token pieces.
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace bl {

class Text {
public:
    explicit Text(const std::string & gguf_path);
    ~Text();
    Text(const Text &) = delete;
    Text & operator=(const Text &) = delete;

    std::vector<int> tokenize(const std::string & s, bool parse_special = true) const;
    std::string      piece(int token) const;   // its bytes (special tokens rendered)
    std::string      detokenize(const std::vector<int> & tokens) const;
    bool             is_eog(int token) const;  // end of generation (<|im_end|>, <|endoftext|>, ...)

    struct Message {
        std::string role;        // system | developer | user | assistant | tool
        std::string content;
        std::string reasoning;   // an assistant turn's thinking (optional)
        std::string tool_calls;  // an assistant turn's calls, already in the template's
                                 // <tool_call><function=...><parameter=...> format (optional)
    };
    // the reasoning effort when thinking: "" = off, "low", "medium", "xhigh"
    // tools_block: the "# Tools ... </IMPORTANT>" system preamble (built by the caller from the request's tools),
    // empty when the request carries no tools; it is put into the system turn as the model's template does.
    static std::string render(const std::vector<Message> & messages, bool add_generation_prompt, const std::string & think,
                              const std::string & tools_block = "");

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

// bytes of token pieces in, complete UTF-8 characters out (a character can span pieces)
class Utf8Stream {
public:
    std::string push(const std::string & bytes);
    std::string flush();   // whatever is left, as is
private:
    std::string pend_;
};

}  // namespace bl
