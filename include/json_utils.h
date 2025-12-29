#ifndef VISION_DETECTOR_JSON_UTILS_H
#define VISION_DETECTOR_JSON_UTILS_H

#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace vision_detector {

// Simple JSON value holder for basic parsing needs
class JsonValue {
public:
    enum Type { Null, Bool, Number, String, Array, Object };

    JsonValue() : type_(Null) {}
    JsonValue(bool b) : type_(Bool), bool_val_(b) {}
    JsonValue(double n) : type_(Number), num_val_(n) {}
    JsonValue(const std::string& s) : type_(String), str_val_(s) {}
    JsonValue(const char* s) : type_(String), str_val_(s) {}

    Type type() const { return type_; }

    bool asBool() const { return bool_val_; }
    double asNumber() const { return num_val_; }
    int asInt() const { return static_cast<int>(num_val_); }
    float asFloat() const { return static_cast<float>(num_val_); }
    const std::string& asString() const { return str_val_; }

    // Object access
    bool has(const std::string& key) const {
        return obj_val_.find(key) != obj_val_.end();
    }

    const JsonValue& operator[](const std::string& key) const {
        static JsonValue null_val;
        auto it = obj_val_.find(key);
        return (it != obj_val_.end()) ? it->second : null_val;
    }

    JsonValue& operator[](const std::string& key) {
        type_ = Object;
        return obj_val_[key];
    }

    // Array access
    size_t size() const {
        if (type_ == Array) return arr_val_.size();
        if (type_ == Object) return obj_val_.size();
        return 0;
    }

    const JsonValue& operator[](size_t idx) const {
        static JsonValue null_val;
        return (idx < arr_val_.size()) ? arr_val_[idx] : null_val;
    }

    void push_back(const JsonValue& val) {
        type_ = Array;
        arr_val_.push_back(val);
    }

    // Object iteration
    const std::map<std::string, JsonValue>& items() const { return obj_val_; }

private:
    Type type_;
    bool bool_val_ = false;
    double num_val_ = 0.0;
    std::string str_val_;
    std::vector<JsonValue> arr_val_;
    std::map<std::string, JsonValue> obj_val_;
};

// Simple JSON parser
class JsonParser {
public:
    static JsonValue parse(const std::string& json) {
        size_t pos = 0;
        return parseValue(json, pos);
    }

    static JsonValue parseFile(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + path);
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        return parse(buffer.str());
    }

private:
    static void skipWhitespace(const std::string& s, size_t& pos) {
        while (pos < s.size() && std::isspace(s[pos])) ++pos;
    }

    static JsonValue parseValue(const std::string& s, size_t& pos) {
        skipWhitespace(s, pos);
        if (pos >= s.size()) return JsonValue();

        char c = s[pos];
        if (c == '"') return parseString(s, pos);
        if (c == '{') return parseObject(s, pos);
        if (c == '[') return parseArray(s, pos);
        if (c == 't' || c == 'f') return parseBool(s, pos);
        if (c == 'n') { pos += 4; return JsonValue(); }  // null
        if (c == '-' || std::isdigit(c)) return parseNumber(s, pos);

        return JsonValue();
    }

    static JsonValue parseString(const std::string& s, size_t& pos) {
        ++pos;  // skip opening quote
        std::string result;
        while (pos < s.size() && s[pos] != '"') {
            if (s[pos] == '\\' && pos + 1 < s.size()) {
                ++pos;
                switch (s[pos]) {
                    case 'n': result += '\n'; break;
                    case 't': result += '\t'; break;
                    case 'r': result += '\r'; break;
                    case '"': result += '"'; break;
                    case '\\': result += '\\'; break;
                    default: result += s[pos]; break;
                }
            } else {
                result += s[pos];
            }
            ++pos;
        }
        ++pos;  // skip closing quote
        return JsonValue(result);
    }

    static JsonValue parseNumber(const std::string& s, size_t& pos) {
        size_t start = pos;
        if (s[pos] == '-') ++pos;
        while (pos < s.size() && (std::isdigit(s[pos]) || s[pos] == '.' || s[pos] == 'e' || s[pos] == 'E' || s[pos] == '+' || s[pos] == '-')) {
            ++pos;
        }
        return JsonValue(std::stod(s.substr(start, pos - start)));
    }

    static JsonValue parseBool(const std::string& s, size_t& pos) {
        if (s.substr(pos, 4) == "true") {
            pos += 4;
            return JsonValue(true);
        }
        pos += 5;  // false
        return JsonValue(false);
    }

    static JsonValue parseArray(const std::string& s, size_t& pos) {
        ++pos;  // skip '['
        JsonValue arr;
        skipWhitespace(s, pos);

        while (pos < s.size() && s[pos] != ']') {
            arr.push_back(parseValue(s, pos));
            skipWhitespace(s, pos);
            if (s[pos] == ',') ++pos;
            skipWhitespace(s, pos);
        }
        ++pos;  // skip ']'
        return arr;
    }

    static JsonValue parseObject(const std::string& s, size_t& pos) {
        ++pos;  // skip '{'
        JsonValue obj;
        skipWhitespace(s, pos);

        while (pos < s.size() && s[pos] != '}') {
            skipWhitespace(s, pos);
            std::string key = parseString(s, pos).asString();
            skipWhitespace(s, pos);
            ++pos;  // skip ':'
            obj[key] = parseValue(s, pos);
            skipWhitespace(s, pos);
            if (s[pos] == ',') ++pos;
            skipWhitespace(s, pos);
        }
        ++pos;  // skip '}'
        return obj;
    }
};

}  // namespace vision_detector

#endif  // VISION_DETECTOR_JSON_UTILS_H