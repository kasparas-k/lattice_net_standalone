#pragma once

#include <iostream>
#include <unordered_map>
#include <math.h>
#include <chrono>
#include <atomic>
#include <vector>
#include <mutex>

#include "scope_exit.h"
#include "Ringbuffer.h"

//for formatting when printing the stats
#include <boost/format.hpp>


#if defined(PROFILER_IMPLEMENTATION) && !defined(PROFILER_HAS_BEEN_IMPLEMENTED)
    #define LOGURU_IMPLEMENTATION 1 //if we are implementing profiler we should also implement the loguru becase we need to use it
#endif
#include "loguru/loguru.hpp"


#ifndef ENABLE_GL_PROFILING
	#define ENABLE_GL_PROFILING 0
#endif

#ifndef ENABLE_CUDA_PROFILING
	#define ENABLE_CUDA_PROFILING 0
#endif



#if ENABLE_GL_PROFILING
    #include <glad/glad.h>
#endif
#if ENABLE_CUDA_PROFILING
    #include <cuda_runtime_api.h>
#endif

namespace radu{
namespace utils{


struct Stats{ //for each timer we store some stats so we can compute the avg, min, max an std-dev https://dsp.stackexchange.com/a/1187

    int nr_samples=0;
    float min=std::numeric_limits<float>::max(); //minimum time taken for that timer
    float max=std::numeric_limits<float>::min(); //maximum time taken for that timer
    float mean=0;
    float exp_mean=-1; //exponential average that weights a bit more the more recent samples
    float variance=0;
    float std_dev=0; //not really necesarry because we have the variance but it's nice to have

    float S=0; //used to calculate the variance and std_dev as explained here https://dsp.stackexchange.com/a/1187

};

inline std::ostream& operator<<(std::ostream& os, const Stats& s){
    os << boost::format("mean: %-7.2f") % s.mean;
    os << boost::format("nr_samples: %-5d") % s.nr_samples;
    os << boost::format("min: %-7.2f") % s.min;
    os << boost::format("max: %-7.2f") % s.max;
    os << boost::format("exp_mean: %-7.2f") % s.exp_mean;
    os << boost::format("variance: %-7.2f") % s.variance;
    os << boost::format("std_dev: %-7.2f") % s.std_dev;

    return os;
}


class Timer{
public:
    //https://stackoverflow.com/a/40136853
    using precision = long double;
    using ratio = std::nano;
    using clock_t=std::chrono::high_resolution_clock;
    using duration_t = std::chrono::duration<precision, ratio>;
    using timepoint_t = std::chrono::time_point<clock_t, duration_t>;


    void start(){
        m_start_time = clock_t::now();
        m_running = true;
    }

    bool stop(){
        //only stop if it was running otherwise it was already stopped before
        if (m_running){
            m_end_time = clock_t::now() +m_duration_other_sections;
            m_duration_other_sections=duration_t::zero();
            m_running = false;
            return true;
        }else{
            return false; //returns that it failed to stop the timer because it was already stopped before
        }
    }

    bool pause(){
        if(stop()){ //if we managed to stop it correctly and it wasn't stopped before
            //if its running we stop the timer and save the time it took until now so we can sum it up the last time we do TIME_END
            m_duration_other_sections  += std::chrono::high_resolution_clock::now()-m_start_time;
            return true;
        }else{
            return false;
        }

    }

    double elapsed_ns(){
        timepoint_t endTime;

        if(m_running){
            endTime = clock_t::now();
        }else{
            endTime = m_end_time;
        }

        return std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - m_start_time).count();
    }

    double elapsed_ms(){
        return elapsed_ns() / 1000000.0;
    }

    double elapsed_s(){
        return elapsed_ms() / 1000.0;
    }

    bool is_running(){
        return m_running;
    }


private:
    timepoint_t m_start_time;
    timepoint_t m_end_time;
    duration_t m_duration_other_sections=duration_t::zero(); //each time you do a pause we accumulate the time it took for that section of start-pause. This will get summed up at the end for the last start-end
    bool m_running = false;
};

//we put it in a namespace so as to not pollute the global namespace
namespace Profiler_ns{
    extern std::unordered_map<std::string, ringbuffer<float,50> > m_timings;  //contains the last N timings of the registers, for plotting in gui
    extern std::unordered_map<std::string, Timer> m_timers;  //contains the timers for the registers
    extern std::vector<std::string> m_ordered_timers;  //contains the timers in the order of insertion, useful for when we plot all of them, they will show in the order we inserted them
    extern std::unordered_map<std::string, Stats > m_stats;
    extern std::mutex m_update_mutex;
    extern bool m_profile_gpu;
    extern bool m_debug_with_profiler;

class Profiler{
public:
    static void start_time( std::string name ){

        if(m_debug_with_profiler){
            LOG(INFO) << "START " << name;
        }

        std::lock_guard<std::mutex> lock(m_update_mutex);  // so that accesed to the map are thread safe

        // std::thread::id thread_id= std::this_thread::get_id();
        // std::stringstream ss;
        // ss << thread_id;
        // std::string thread_id_string = ss.str();
        // std::string full_name=name+ thread_id_string;
        std::string full_name=name;

        m_timers[full_name].start();

    }

    static void stop_time(std::string name){

        if(m_debug_with_profiler){
            LOG(INFO) << "STOP " << name;
        }

        std::lock_guard<std::mutex> lock(m_update_mutex);  // so that accesed to the map are thread safe

        // std::thread::id thread_id= std::this_thread::get_id();
        // std::stringstream ss;
        // ss << thread_id;
        // std::string thread_id_string = ss.str();
        // std::string full_name=name+ thread_id_string;
        std::string full_name=name;

        //it's the first time we stopped this timer and doesn't have any recordings yet
        if(m_timings[full_name].empty()){
            m_ordered_timers.push_back(full_name);

            //the first time we time a functions it's usualy a warm up so the maximum can be quite big. We ignore this one so as to not squeue our stats
            m_timers[full_name].stop();
            double time_elapsed=m_timers[full_name].elapsed_ms();
            // VLOG(1) << "Time elapsed for " << full_name << " " << time_elapsed;
            m_timings[full_name].push(time_elapsed);
            return; //don't store any stats, because this is the first time we time this function so it's likely to be bad
        }

        //get elapsed time for that timer and register it into the timings
        if(m_timers[full_name].stop()){  //we manage to stop is correctly and it was no stopped before
            double time_elapsed=m_timers[full_name].elapsed_ms();
            m_timings[full_name].push(time_elapsed);

            //we also store some evaluation things to be able to easily calculate min, max and std-dev
            //https://dsp.stackexchange.com/a/1187
            Stats& stats=m_stats[full_name];
            stats.nr_samples+=1;
            float prev_mean=stats.mean;
            stats.mean= stats.mean + (time_elapsed-stats.mean)/stats.nr_samples;
            float alpha=0.07;
            if(stats.exp_mean==-1){ // it's the first time we add something to the exp_mean
                stats.exp_mean= time_elapsed;
            }else{
                stats.exp_mean= stats.exp_mean + alpha*(time_elapsed-stats.exp_mean);
            }
            stats.S=stats.S+ (time_elapsed - stats.mean)*(time_elapsed-prev_mean);
            if(stats.nr_samples>1){ //to avoid division by zero
                stats.std_dev=sqrt( stats.S/ (stats.nr_samples-1) );
                stats.variance=stats.S/ (stats.nr_samples-1);
            }
            if(time_elapsed < stats.min){
                stats.min=time_elapsed;
            }
            if(time_elapsed > stats.max){
                stats.max=time_elapsed;
            }

        }

    }


    static void pause_time( std::string name ){

        if(m_debug_with_profiler){
            LOG(INFO) << "PAUSE " << name;
        }

        std::lock_guard<std::mutex> lock(m_update_mutex);  // so that accesed to the map are thread safe
        // std::thread::id thread_id= std::this_thread::get_id();
        // std::stringstream ss;
        // ss << thread_id;
        // std::string thread_id_string = ss.str();
        // std::string full_name=name+ thread_id_string;
        std::string full_name=name;

        m_timers[full_name].pause();

    }

    static void print_all_stats(){

        std::lock_guard<std::mutex> lock(m_update_mutex);  // so that accesed to the map are thread safe
        std::cout << "====STATS==="<< std::endl;
        //goes through all the timers and prints all the stats
        for (size_t i = 0; i < m_ordered_timers.size(); i++) {
            std::string name=m_ordered_timers[i];
            // std::cout << "\t name: " << name << m_stats[name] << std::endl;
            std::cout << boost::format("name: %-30s") % name << m_stats[name] << std::endl;
        }
    }

    //getters
    static std::unordered_map<std::string, ringbuffer<float,50> > timings(){
        return m_timings;
    }
    static std::unordered_map<std::string, Timer> timers(){
        return m_timers;
    }
    static std::vector<std::string> ordered_timers(){
        return m_ordered_timers;
    }
    static std::unordered_map<std::string, Stats > stats(){
        return m_stats;
    }
    static double get_elapsed_ms(const std::string timer_name){
        std::lock_guard<std::mutex> lock(m_update_mutex);
        double elapsed_ms=m_timers[timer_name].elapsed_ms();
        return elapsed_ms;
    }
    static double get_elapsed_ns(const std::string timer_name){
        std::lock_guard<std::mutex> lock(m_update_mutex);
        double elapsed_ns=m_timers[timer_name].elapsed_ns();
        return elapsed_ns;
    }
    static void clear(){
        std::lock_guard<std::mutex> lock(m_update_mutex);
        m_timings.clear();
        m_timers.clear();
        m_ordered_timers.clear();
        m_stats.clear();
    }


private:
    Profiler();
};


//does sync of gpu if we are compiling with certain gpu and if we are profiling the gpu
//Needs to be inline as this will be different for each cxx files depending if they declare GL or CUDA profiling
inline void sync_gpu(){
    if(m_profile_gpu){
        #if ENABLE_GL_PROFILING
            glFinish();
        #endif
        #if ENABLE_CUDA_PROFILING
            cudaDeviceSynchronize();
        #endif
    }
}
inline static bool is_profiling_gpu(){
    return m_profile_gpu;
}
inline static void set_profile_gpu(const bool val){
    m_profile_gpu=val;
}

} //namespace Profiler_ns



//MACROS
#define TIME_START(name) \
    radu::utils::Profiler_ns::sync_gpu();\
    radu::utils::Profiler_ns::Profiler::start_time(name);
#define TIME_END(name) \
    radu::utils::Profiler_ns::sync_gpu();\
    radu::utils::Profiler_ns::Profiler::stop_time(name);
//when you have to sections that are disjoin but you want to get the time it take for both, you can start-pause start-end
#define TIME_PAUSE(name) \
    radu::utils::Profiler_ns::sync_gpu();\
    radu::utils::Profiler_ns::Profiler::pause_time(name);
#define TIME_SCOPE(name)\
    radu::utils::Profiler_ns::sync_gpu();\
    radu::utils::Profiler_ns::Profiler::start_time(name); \
    SCOPE_EXIT{radu::utils::Profiler_ns::sync_gpu(); radu::utils::Profiler_ns::Profiler::stop_time(name);};
#define PROFILER_PRINT()\
    radu::utils::Profiler_ns::Profiler::print_all_stats();
inline float ELAPSED(std::string name){
    //check if the name is in the timings
    auto got = radu::utils::Profiler_ns::m_timings.find (name);
    if ( got == radu::utils::Profiler_ns::m_timings.end() ){
        LOG(FATAL) << "There are no timings recorded for name " <<name;
    }
    float last_timing=got->second.back();

    return last_timing;
}




// ----------------------------------------------------------------------------
// 88 8b    d8 88""Yb 88     888888 8b    d8 888888 88b 88 888888    db    888888 88  dP"Yb  88b 88
// 88 88b  d88 88__dP 88     88__   88b  d88 88__   88Yb88   88     dPYb     88   88 dP   Yb 88Yb88
// 88 88YbdP88 88"""  88  .o 88""   88YbdP88 88""   88 Y88   88    dP__Yb    88   88 Yb   dP 88 Y88
// 88 88 YY 88 88     88ood8 888888 88 YY 88 888888 88  Y8   88   dP""""Yb   88   88  YbodP  88  Y8

/* In one of your .cpp files you need to do the following:
#define PROFILER_IMPLEMENTATION 1
#include <Profiler.h>

This will define all the Profiler functions so that the linker may find them.
*/

#if defined(PROFILER_IMPLEMENTATION) && !defined(PROFILER_HAS_BEEN_IMPLEMENTED)
#define PROFILER_HAS_BEEN_IMPLEMENTED


//we put it in a namespace so as to not pollute the global namespace
namespace Profiler_ns{
    std::unordered_map<std::string, ringbuffer<float,50> > m_timings;  //contains the last N timings of the registers, for plotting in gui
    std::unordered_map<std::string, Timer> m_timers;  //contains the timers for the registers
    std::vector<std::string> m_ordered_timers;  //contains the timers in the order of insertion, useful for when we plot all of them, they will show in the order we inserted them
    std::unordered_map<std::string, Stats > m_stats;
    std::mutex m_update_mutex; // when adding a new measurement to the maps, we need to lock them so it can be thread safe
    bool m_profile_gpu=false;
    bool m_debug_with_profiler=false;
    //whatever you addd here needs to be also added as extern at the start of the class
}



#endif // PROFILER_IMPLEMENTATION

} //namespace utils
} //namespace radu
