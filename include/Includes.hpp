#pragma once
#include<SFML/Graphics.hpp>
#include<chrono>
//#define CLOSE_BOOST

//c++26 when
constexpr double pow10(const double& x) {
	if (x >= 0) return x + 1;
	else if (x >= -1) return 0.9999884 + 2.300028 * x + 2.618158 * x * x + 1.884004 * x * x * x + 0.8459377 * x * x * x * x + 0.180063 * x * x * x * x * x;
	else if (x >= -2) return  0.8828179 + 1.747242 * x + 1.514643 * x * x + 0.7068385 * x * x * x + 0.1746253 * x * x * x * x + 0.0180063 * x * x * x * x * x;
	else if (x >= -3) return  0.5044173 + 0.7685575 * x + 0.4862973 * x * x + 0.1585403 * x * x * x + 0.02646568 * x * x * x * x + 0.00180063 * x * x * x * x * x;
	else if (x >= -4) return  0.1946079 + 0.2331639 * x + 0.1138718 * x * x + 0.02824093 * x * x * x + 0.003546883 * x * x * x * x + 0.000180063 * x * x * x * x * x;
	else if (x >= -5) return  0.05736114 + 0.05607182 * x + 0.02216765 * x * x + 0.004422909 * x * x * x + 0.0004447198 * x * x * x * x + 0.0000180063 * x * x * x * x * x;
	else if (x >= -6) return  0.01404862 + 0.01155448 * x + 0.003828476 * x * x + 0.0006381851 * x * x * x + 0.00005347513 * x * x * x * x + 0.00000180063 * x * x * x * x * x;
	else if (x >= -7) return  0.003012504 + 0.002134889 * x + 0.0006081889 * x * x + 0.00008700919 * x * x * x + 0.000006247828 * x * x * x * x + 1.80063e-7 * x * x * x * x * x;
	return  0.0005849019 + 0.0003638186 * x + 0.0000908504 * x * x + 0.00001138011 * x * x * x + 7.148143e-7 * x * x * x * x + 1.80063e-8 * x * x * x * x * x;
}

constexpr long long particles = 2048*8;
constexpr long long updates_per_frame = 1;
constexpr double radius = 2048*16;
constexpr double minerror = pow10(-1.6); //-1, -1.2 rips, 1.6 galaxy,-1.9 no galaxies
constexpr double probabilistic_gravity_constant = minerror*minerror;
constexpr double classic_gravity_constant = probabilistic_gravity_constant;
constexpr long long min_delay_error = 1000000;
constexpr unsigned long long nodes_reserved = particles*3;
constexpr uint_fast32_t threads = 8u;
constexpr uint_fast32_t section_count = threads;
constexpr uint_fast32_t min_section_length = 64u;

struct MyClock {
	std::chrono::high_resolution_clock::time_point starting_time = std::chrono::high_resolution_clock::now();

	auto elapsed_nanoseconds() {
		return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - starting_time).count();
	}
	double elapsed_seconds() {
		return (std::chrono::high_resolution_clock::now() - starting_time).count()/1.e9;
	}

	void restart() {
		starting_time = std::chrono::high_resolution_clock::now();
	}

	static constexpr unsigned long long getPartsPerSecond() {
		return 1000000000ull;
	}

	static void sleep_seconds(double const& delay) {
		if (delay <= 0) return;
		std::this_thread::sleep_for(std::chrono::duration<double, std::ratio<1, 1>>(delay));
		return;
	}

	static void sleep_nanoseconds(long long const& delay) {
		if (delay <= 0) return;
		std::this_thread::sleep_for(std::chrono::duration<long long, std::nano>(delay));
		return;
	}

	static long long sleep_for_how_long(long long const& delay) {
		static MyClock clock;
		clock.restart();
		sleep_nanoseconds(delay);
		return clock.elapsed_nanoseconds();
	}
};

struct DelayManager {
	long long max_extra_delay = min_delay_error;
	void delay_nanoseconds(long long const& target_delay) {
		if (target_delay <= max_extra_delay) return;
		long long real_delay = MyClock::sleep_for_how_long(target_delay - max_extra_delay);
		if (real_delay > target_delay) {
			max_extra_delay = real_delay - target_delay + max_extra_delay;
		}
	}

	void reset() {
		max_extra_delay = min_delay_error;
	};
};



typedef sf::Vector2<double> Vector2d;
template<class T>
using container_type = std::vector<T>;
using namespace std::literals;

template<typename f>
constexpr f distance(const sf::Vector2<f> v, const sf::Vector2<f> w) {
	const f dx = w.x - v.x;
	const f dy = w.y - v.y;
	return std::sqrt(dx * dx + dy * dy);
}

template<typename f>
constexpr f distance_squared(const sf::Vector2<f> v, const sf::Vector2<f> w) {
	const f dx = w.x - v.x;
	const f dy = w.y - v.y;
	return dx * dx + dy * dy;
}


template<typename f>
constexpr f distance_squared_error(const sf::Vector2<f> v, const sf::Vector2<f> w) {
	const f dx = w.x - v.x;
	const f dy = w.y - v.y;
	return dx * dx + dy * dy + minerror;
}

template<typename f>
constexpr sf::Vector2<f> inverse_square_on_second_nobranch(const sf::Vector2<f> v, const sf::Vector2<f> w) {
	const f dx = w.x - v.x;
	const f dy = w.y - v.y;
	const f distance = std::sqrt(dx * dx + dy * dy + minerror);
	const f cubed = distance * distance * distance;
	return { dx / cubed, dy / cubed };
}

template<typename f>
constexpr sf::Vector2<f> inverse_square_on_second_nobranch(const sf::Vector2<f> v, const sf::Vector2<f> w, const f distance_squared) {
	const f dx = w.x - v.x;
	const f dy = w.y - v.y;
	const f distance = std::sqrt(distance_squared);
	const f cubed = distance * distance * distance;
	return { dx / cubed, dy / cubed };
}

template<typename f, typename d>
constexpr sf::Vector2<f> normalize_on_second(const sf::Vector2<f> v, const sf::Vector2<f> w, d dist) {
	const f dx = w.x - v.x;
	const f dy = w.y - v.y;
	return { static_cast<f>(dx / dist), static_cast<f>(dy / dist) };
}