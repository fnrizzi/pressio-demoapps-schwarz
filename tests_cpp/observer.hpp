
#ifndef PRESSIODEMOAPPS_TESTS_OBSERVER_HPP_
#define PRESSIODEMOAPPS_TESTS_OBSERVER_HPP_

template <typename StateType>
class FomObserver
{
public:
    FomObserver(const std::string & f0, int freq)
        : myfile0_(f0,  std::ios::out | std::ios::binary),
        sampleFreq_(freq){}

    ~FomObserver(){
        myfile0_.close();
    }

    FomObserver() = default;
    FomObserver & operator=(FomObserver &) = default;
    FomObserver & operator=(FomObserver &&) = default;

    template<typename TimeType>
    void operator()(const pressio::ode::StepCount stepIn,
            const TimeType /*timein*/,
            const StateType & state)
    {
        const auto step = stepIn.get();
        if (step % sampleFreq_ == 0){
        const std::size_t ext = state.size()*sizeof(double);
        myfile0_.write(reinterpret_cast<const char*>(&state(0)), ext);
        }
    }

private:
    std::ofstream myfile0_;
    int sampleFreq_ = {};
};

class StateObserver
{
public:
    StateObserver(const std::string & f0, int freq)
        : myfile_(f0,  std::ios::out | std::ios::binary),
        sampleFreq_(freq){}

    explicit StateObserver(int freq)
        : myfile_("state_snapshots.bin",  std::ios::out | std::ios::binary),
        sampleFreq_(freq){}

    ~StateObserver(){ myfile_.close(); }

    template<typename TimeType, typename ObservableType>
    std::enable_if_t< pressio::is_vector_eigen<ObservableType>::value >
    operator()(pressio::ode::StepCount step,
            const TimeType /*timeIn*/,
            const ObservableType & state)
    {
        if (step.get() % sampleFreq_ == 0){
        const std::size_t ext = state.size()*sizeof(typename ObservableType::Scalar);
        myfile_.write(reinterpret_cast<const char*>(&state(0)), ext);
        }
    }

private:
    std::ofstream myfile_;
    const int sampleFreq_ = {};
};

// This can really only be used with the Schwarz framework,
//      can't be used with the original PDA stepper
class RuntimeObserver
{
    public:
        RuntimeObserver(const std::string & f0, int nDomains_in = 1)
            : timeFile_(f0, std::ios::out | std::ios::binary)
            , nDomains_(static_cast<std::size_t>(nDomains_in))
        {
            // 8-byte file header indicating number of domains
            timeFile_.write(reinterpret_cast<const char*>(&nDomains_), sizeof(std::size_t));
        }

        ~RuntimeObserver() { timeFile_.close(); }

        void operator() (std::vector<std::vector<double>> & runtimeVec)
        {
            // runtimeVec has dimensions (nDomains, nIters)
            // record all dimensions with header indicating number of iterations

            // 8-byte record header indicating number of subiterations in this iteration
            std::size_t niters = runtimeVec[0].size();
            timeFile_.write(reinterpret_cast<const char*>(&niters), sizeof(std::size_t));

            // unroll iterations, then domain order
            for (int iterIdx = 0; iterIdx < niters; ++iterIdx) {
                for (int domIdx = 0; domIdx < nDomains_; ++domIdx) {
                    timeFile_.write(reinterpret_cast<const char*>(&runtimeVec[domIdx][iterIdx]), sizeof(double));
                }
            }

        }

        void operator() (double runtimeVal)
        {
            // Overload for handling non-Schwarz input, have to manually time it and pass total runtime
            // Just write it as if there's one iteration, one domain
            std::size_t one = static_cast<std::size_t>(1);
            timeFile_.write(reinterpret_cast<const char*>(&one), sizeof(std::size_t));
            timeFile_.write(reinterpret_cast<const char*>(&runtimeVal), sizeof(double));
        }

    private:
        std::ofstream timeFile_;
        std::size_t nDomains_;
};

#endif
