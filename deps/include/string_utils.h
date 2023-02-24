#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>



// //loguru
// #include <loguru.hpp>

namespace radu{
namespace utils{



// trim from start (in place)
static inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
            std::not1(std::ptr_fun<int, int>(std::isspace))));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(),
            std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}

// trim from start (copying)
static inline std::string ltrim_copy(std::string s) {
    ltrim(s);
    return s;
}

// trim from end (copying)
static inline std::string rtrim_copy(std::string s) {
    rtrim(s);
    return s;
}

// trim from both ends (copying)
static inline std::string trim_copy(std::string s) {
    trim(s);
    return s;
}

static inline bool contains(const std::string line, const std::string query_word){
    if (line.find(query_word) != std::string::npos) {
        return true;
    }
    return false;
}

//https://stackoverflow.com/a/37454181
inline std::vector<std::string> split(const std::string& str, const std::string& delim){
    std::vector<std::string> tokens;
    size_t prev = 0, pos = 0;
    do{
        pos = str.find(delim, prev);
        if (pos == std::string::npos) pos = str.length();
        std::string token = str.substr(prev, pos-prev);
        if (!token.empty()) tokens.push_back(token);
        prev = pos + delim.length();
    }
    while (pos < str.length() && prev < str.length());
    return tokens;
}

//joins the tokens together in a string separated by a certain delimiter  https://stackoverflow.com/a/5289170
template <typename Range, typename Value = typename Range::value_type>
inline std::string join(Range const& tokens, const std::string& delim){
    std::ostringstream os;
    auto b = std::begin(tokens), e = std::end(tokens);

    if (b != e) {
        std::copy(b, std::prev(e), std::ostream_iterator<Value>(os, delim.c_str() ));
        b = std::prev(e);
    }
    if (b != e) {
        os << *b;
    }

    return os.str();
}



// Erase all Occurrences of given substring from main string
inline std::string erase_substring(const std::string& main_string, const std::string& to_erase){
    std::string string_modified=main_string;
    size_t pos = std::string::npos;
    // Search for the substring in string in a loop untill nothing is found
    while ((pos  = string_modified.find(to_erase) )!= std::string::npos){
        // If found then erase it from string
        string_modified.erase(pos, to_erase.length());
    }

    return string_modified;
}
//https://thispointer.com/how-to-remove-substrings-from-a-string-in-c/#:~:text=Remove%20First%20Occurrence%20of%20given%20substring%20from%20main%20string&text=*%20Erase%20First%20Occurrence%20of%20given%20substring%20from%20main%20string.&text=To%20remove%20first%20occurrence%20of,remove%20it%20from%20string%20i.e.
inline std::string erase_substrings(std::string & main_string, const std::vector<std::string> & to_erase_list){

    std::string string_modified=main_string;
    for(size_t i=0; i<to_erase_list.size(); i++){
        string_modified= erase_substring(string_modified, to_erase_list[i]);
    }

    return string_modified;
}

inline std::string lowercase(const std::string str){
    std::string new_str=str;
    std::transform(new_str.begin(), new_str.end(), new_str.begin(), ::tolower);
    return new_str;
}

inline std::string uppercase(const std::string str){
    std::string new_str=str;
    std::transform(new_str.begin(), new_str.end(), new_str.begin(), ::toupper);
    return new_str;
}

//https://stackoverflow.com/questions/7276826/c-format-number-with-commas
template<class T>
std::string format_with_commas(T value) {
    std::stringstream ss;
    ss.imbue(std::locale(""));
    ss << std::fixed << value;
    return ss.str();
}

inline std::string file_to_string (const std::string &filename){
    std::ifstream t(filename);
    if (!t.is_open()){
        // LOG(FATAL) << "Cannot open file " << filename;
        throw std::runtime_error( "Cannot open file " + filename );
    }
    return std::string((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
}



} //namespace utils
} //namespace radu
