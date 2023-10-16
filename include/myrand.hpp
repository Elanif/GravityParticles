#pragma once
#include<string>
#include<fstream>
#include<array>
#include<random>

bool saveArray(const double* pdata, size_t length, const std::string& file_path)
{
    std::ofstream os(file_path.c_str(), std::ios::binary | std::ios::out);
    if (!os.is_open())
        return false;
    os.write(reinterpret_cast<const char*>(pdata), std::streamsize(length * sizeof(double)));
    os.close();
    return true;
}

bool loadArray(double* pdata, size_t length, const std::string& file_path)
{
    std::ifstream is(file_path.c_str(), std::ios::binary | std::ios::in);
    if (!is.is_open())
        return false;
    is.read(reinterpret_cast<char*>(pdata), std::streamsize(length * sizeof(double)));
    is.close();
    return true;
}

template<std::size_t N>
std::array<double, N> get_array(const std::string& file_path) {
    std::array<double, N> result;
    loadArray(result.data(), N, file_path);
    return result;
}

const std::array<double, 256> arr256=get_array<256u>("array256.bin");
const std::array<double, 65536> arr65536 =get_array<65536u>("array65536.bin");

//2nd fastest
struct rand256 {
    uint8_t state = 0;
    const double& operator()() {
        return arr256[++state];
    }
};

//fastest
struct rand65536 {
    uint16_t state = 0;
    const double& operator()() {
        return arr65536[++state];
    }
    using result_type = double;
    static constexpr double min = 0;
    static constexpr double max = 1;
};

struct xorshift32 {
    uint64_t state = 1;
    uint32_t operator()()
    {
        /* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return static_cast<uint32_t>(state);
    }

};

//slower then minstd, but when used with normalizer it's faster?
struct xorshift64 {
    uint64_t state=1;
    uint64_t operator()()
    {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return  state;
    }
};

//slightly slower than xorshift64
struct xorshift64s {
    int64_t state = 1;
    uint64_t operator()()
    {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;

        return state * 2685821657736338717ull;
    }
};

//incredibly fast normalizer
template<class RandomEngine64>
struct rand_normalizer64 {
    RandomEngine64 minstdrand;
    union {
        double d;
        long long l;
    } x;
    double operator()() {
        x.d = 1.0;
        x.l |= minstdrand() & (1LL << 53) - 1;
        return x.d - 1;
    }
};

template<class RandomEngine32>
struct rand_normalizer32 {
    RandomEngine32 minstdrand;
    double operator()() {
        x.d = 1.0;
        x.l |= (static_cast<int64_t>(minstdrand()) << 32) & (1LL << 53) - 1;
        return x.d - 1;
    }
private:
    union {
        double d;
        long long l;
    } x;
};

//tldr array is fastest
//2nd fastest and also more rng safe: rand_normalizer64<xorshift64>, pity that i can't use the fastest lehmer thing with __uint128_t

/*{
    const std::size_t N = 256;
    std::random_device rd;
    std::uniform_real_distribution<double> uni(0., 1.);
    std::array<double, N> output;
    for (std::size_t i = 0; i < N; ++i) {
        output[i] = uni(rd);
    }
    saveArray(output.data(), N, std::string("array")+std::to_string(N) + std::string(".bin"));
}*/