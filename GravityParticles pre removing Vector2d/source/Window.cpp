#include"Includes.hpp"
#include"Window.hpp"
#include<iostream>
#include"Particles.hpp"
#include<chrono>
#include<random>
#include"myrand.hpp"

Window::Window(const unsigned int& _width, const unsigned int& _height, std::string const& _title)
:	width(_width),
	height(_height),
	title(_title)
{
	window.create(sf::VideoMode(width, height), title);
	window.clear(sf::Color::Black);

	window.setActive(false);

	std::thread	thread(&Window::render_thread, this);

	sf::Event event;

	while (isWindowOpen() && !close_window.load()) {
		if (window.waitEvent(event)) {
			event_queue.push(event);
		}
	}
	if (thread.joinable())
		thread.join();
	window.setActive(true);
	window.close();
}

bool Window::isWindowOpen()
{
	std::lock_guard<std::mutex> fullscreen_lock(window_mutex);
	return window.isOpen();
}

void Window::render_thread() {

	window.setActive();

	GravityParticles gp(sf::Points, particles, type::probabilistic);
	if (particles==2)
		gp.perfect_setup();
	//gp.start_work();


	double frames = 0;

	sf::View view(sf::Vector2f(0, 0), sf::Vector2f(radius*2.5, radius*2.5));

	grid_container gc(16, 3, 3);
	gc.precalc_grid_ratio(-2000, 2000, -2000, 2000);

	std::chrono::high_resolution_clock c;
	window.setView(view);
	std::chrono::time_point t = c.now();

	sf::Vector2f old_window_size{ 1000,1000 };
	float old_window_ratio = old_window_size.y / old_window_size.x;


	float view_multi = 1;
	sf::FloatRect view_rect;

	gp.update_first();

	MyClock fps_clock;
	DelayManager delay_manager;
	long long time_per_frame = 1000000000 / 60;

	while (!close_window) {
		while (auto event = event_queue.pop_if_not_empty())
		{
			switch (event->type) {
			case sf::Event::Closed:
				close_window.store(true);
				break;
			case sf::Event::KeyPressed: {
				sf::Vector2f view_size = view.getSize();
				sf::Vector2f view_center = view.getCenter();
				if (event->key.code == sf::Keyboard::A) {
					view.setSize({ view_size.x * 1.25f,view_size.y * 1.25f });
				}
				if (event->key.code == sf::Keyboard::S) {
					view.setSize({ view_size.x / 1.25f,view_size.y / 1.25f });
				}
				if (event->key.code == sf::Keyboard::Left) {
					view_center.x -= view_size.x / 4.f;
				}
				if (event->key.code == sf::Keyboard::Right) {
					view_center.x += view_size.x / 4.f;
				}
				if (event->key.code == sf::Keyboard::Up) {
					view_center.y -= view_size.y / 4.f;
				}
				if (event->key.code == sf::Keyboard::Down) {
					view_center.y += view_size.y / 4.f;
				}
				view.setCenter(view_center);
				window.setView(view);
			}
									  break;
			case sf::Event::Resized: {
				sf::Vector2f new_view = sf::Vector2f{ static_cast<float>(event->size.width),static_cast<float>(event->size.width) };
			}
				break;
			case sf::Event::MouseWheelScrolled:
			{
				if (event->mouseWheel.y > 0) {
					sf::Vector2f view_size = view.getSize();
					view.setSize({ view_size.x * 2,view_size.y * 2 });

					window.setView(view);
				}
			}
			break;
			}
		}


		if (fps_clock.elapsed_nanoseconds() < time_per_frame) {
			delay_manager.delay_nanoseconds(time_per_frame - fps_clock.elapsed_nanoseconds());
			continue;
		}
		fps_clock.restart();

		++frames;
		if (static_cast<int>(frames) % 120 == 0) {
			//bad for multithreading
			int horiz_length = view.getSize().x;
			std::stringstream ss;
			ss << "Fps= " << (std::chrono::seconds(1) * (frames) / (c.now() - t))<<", horiz scale: "<<horiz_length<<" px";
			std::cout << ss.str() << "\n";
			/*gp.print_speed();
			std::cout<< "\n";*/
			/*sf::Text fps_text;
			fps_text.setFillColor(sf::Color::White);
			fps_text.setString(ss.str());
			fps_text.setCharacterSize(1999);
			window.draw(fps_text);*/
		}
		window.clear();
		window.draw(gp);

		window.display();

		for (std::size_t i = 0; i < updates_per_frame; ++i) {
			//gp.update_multithreading();
			//gp.update_multithreading_pre2();
			//gp.update_multithreading_pre2_norand();
			//gp.continue_work();
			gp.update();
		}
	}
	gp.stop_work();
}