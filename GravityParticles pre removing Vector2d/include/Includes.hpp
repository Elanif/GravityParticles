#pragma once
#include<SFML/Graphics.hpp>
#include<chrono>
//#define CLOSE_BOOST
constexpr long long particles = 512;
constexpr long long updates_per_frame = 64;
constexpr double radius = 400;
constexpr double minerror = 5e-2;
constexpr double gravity_constant = minerror*minerror;
constexpr long long min_delay_error = 1000000;

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