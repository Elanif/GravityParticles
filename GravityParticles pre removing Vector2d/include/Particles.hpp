#pragma once
#include"Includes.hpp"
#include<SFML/Graphics.hpp>
#include<cmath>
#include<numbers>
#include<random>
#include<iostream>
#include<iomanip>
#include<vector>
#include<array>
#include<algorithm>
#include<thread>
#include<SafeContainers.hpp>
#include<initializer_list>
#include<memory>
#include"myrand.hpp"
#include<list>

typedef sf::Vector2<double> Vector2d;

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
    return dx * dx + dy * dy+minerror;
}

template<typename f>
constexpr sf::Vector2<f> inverse_square_on_second_nobranch(const sf::Vector2<f> v, const sf::Vector2<f> w) {
    const f dx = w.x - v.x;
    const f dy = w.y - v.y;
    const f distance = std::sqrt(dx * dx + dy * dy + minerror);
    const f cubed = distance * distance * distance;
    return { dx / cubed, dy / cubed };
}

template<typename f, typename d>
constexpr sf::Vector2<f> normalize_on_second(const sf::Vector2<f> v, const sf::Vector2<f> w, d dist) {
    const f dx = w.x - v.x;
    const f dy = w.y - v.y;
    return { static_cast<f>(dx / dist), static_cast<f>(dy / dist) };
}

struct Point {
    Vector2d& position;
    Vector2d& speed;
    Vector2d& accel;
    const double mass{ 1 };
    Point(Vector2d& position, Vector2d& speed, Vector2d& accel, double mass)
        :position(position), speed(speed), accel(accel), mass(mass) {};
};

enum class type {
    classic,
    probabilistic
};

constexpr std::vector<std::vector<std::pair<std::size_t, std::size_t>>> round_robin(std::size_t teams) {
    std::size_t rounds = teams - 1;
    std::size_t mpr = teams / 2;

    std::vector<std::size_t> t(teams);
    for (std::size_t i = 0; i < teams; ++i) {
        t[i] = i;
    }

    std::vector<std::vector<std::pair<std::size_t, std::size_t>>> matches(rounds, std::vector<std::pair<std::size_t, std::size_t>>(mpr));
    for (std::size_t r = 0; r < rounds; ++r) {
        for (std::size_t m = 0; m < mpr; ++m) {
            matches[r][m] = std::make_pair(t[m], t[teams - m -1]);
        }
        t.erase(std::find(t.begin(), t.end(), rounds - r ));
        t.insert(t.begin() + 1, rounds - r);
    }

    matches.push_back({});
    matches[matches.size() - 1].resize(teams);
    for (std::size_t i = 0; i < teams; ++i) {
        matches[matches.size() - 1][i] = std::pair<std::size_t, std::size_t> (i,i);
    }
    return matches;
}

template<typename contained>
struct grid {
    grid() {};

    grid(uint_fast32_t width, uint_fast32_t height) :
        width(width), height(height)
    {};
    std::vector<contained> grid_flattened;
    void set_padding(uint_fast32_t width_padding, uint_fast32_t height_padding) {
        this->width_padding = width_padding;
        this->height_padding = height_padding;
        this->total_width = width+2* width_padding;
        this->total_height = height +2* height_padding;
    }
    void resize_vector(uint_fast64_t size) {
        grid_flattened.resize(size, {});
    }
    void resize_vector() {
        resize_vector(static_cast<uint_fast64_t>(total_width) * total_height);
    }
    void precalc(double left, double right, double top, double bottom) {
        this->top = top;
        this->bottom = bottom;
        this->left = left;
        this->right = right;
        cell_width_inv = width / (right - left);
        cell_width = 1./cell_width_inv;
        cell_height_inv = height / (bottom - top);
        cell_height = 1. / cell_height_inv;
    }
    uint_fast32_t width=1;
    uint_fast32_t total_width=1;
    uint_fast32_t width_padding=1;
    uint_fast32_t height=1;
    uint_fast32_t total_height=1;
    uint_fast32_t height_padding=1;
    double top{ 0 };
    double bottom{ 0 };
    double left{ 0 };
    double right{ 0 };
    double cell_width_inv{ 0 };
    double cell_width{ 0 };
    double cell_height_inv{ 0 };
    double cell_height{ 0 };
    void clear() {
        grid_flattened.clear();
    }
    contained& operator()(uint_fast32_t const& i, uint_fast32_t const& j) {
        return grid_flattened[i + j * total_width];
    }
    contained& operator()(double const& x, double const& y) {
        auto pos = get_coords(x, y);
        return this->operator()(pos.first, pos.second);
    }
    contained& operator()(Vector2d const& v) {
        return this->operator()(v.x, v.y);
    }
    std::pair<uint_fast32_t, uint_fast32_t> get_coords(double x, double y) {
        std::pair<uint_fast32_t, uint_fast32_t> result;
        result.first = (x - left) * cell_width_inv + width_padding;
        result.second = (y - top) * cell_height_inv + height_padding;
        return result;
    }
    Vector2d get_center(uint_fast32_t const& i, uint_fast32_t const& j) {
        return {(i+0.5)*cell_width+left,(j+0.5)*cell_height+right}; //todo add padding correction
    }
};

template<template<class> typename container>
struct grid_index_container: public grid<container<std::size_t> >{

    //maybe it's this?
    using grid<container<std::size_t> >::grid;
    std::vector<double> grid_flattened_weights;
    double& weight(double const& x, double const& y) {
        auto pos = this->get_coords(x, y);
        return weight(pos.first, pos.second);
    }
    double& weight(Vector2d const& v) {
        return weight(v.x, v.y);
    }
    double& weight( uint_fast32_t const& i, uint_fast32_t const& j) {
        return grid_flattened_weights[i + j * this->total_width];
    }
    void resize_vector() {
        grid<container<std::size_t> >::resize_vector();
        grid_flattened_weights.resize(static_cast<uint_fast64_t>(this->total_width) * this->total_height, {});
    }
    container<std::size_t> top_left;
};

struct grid_container {
    Vector2d random_grid_offset;
    double top{ 0 };
    double bottom{ 0 };
    double left{ 0 };
    double right{ 0 };
    const uint_fast32_t base_grid_size = 16;
    const uint_fast32_t grid_depth = 4;
    const uint_fast32_t grid_division = 3;
    uint_fast32_t finest_width = 1;
    uint_fast32_t finest_height = 1;
    grid_index_container<std::list> grid_points;
    std::vector<grid<double> > grid_weights_vector;

    grid_container(uint_fast32_t base_grid_size, uint_fast32_t grid_depth, uint_fast32_t grid_subdivision) :
        base_grid_size(base_grid_size),
        grid_depth(grid_depth),
        grid_division(grid_subdivision)
    {
        grid_weights_vector.resize(grid_depth - 1);
        uint_fast32_t grid_size = base_grid_size;
        for (uint_fast32_t i = 0; i < grid_depth - 1; ++i, grid_size *= grid_subdivision) {
            grid_weights_vector[i].width = grid_size;
            grid_weights_vector[i].height = grid_size;
            grid_weights_vector[i].set_padding(grid_depth - i, grid_depth - i);
            grid_weights_vector[i].resize_vector();
        }
        grid_points.width = grid_size;
        grid_points.height = grid_size;
        grid_points.set_padding(1,1);
        grid_points.resize_vector();
    }

    void precalc_grid_ratio(double left, double right, double top, double bottom) {
        for (auto& g : grid_weights_vector) {
            g.precalc(left, right, top, bottom);
        }
        grid_points.precalc(left, right, top, bottom);
        double grid_ratio = base_grid_size;
    }
    void add(const Point& p, const std::size_t& point_index) {
        add(p.position, p.mass, point_index);
    }
    void add(const Vector2d& position, double const& mass, const std::size_t& point_index) {
        for (uint_fast32_t i = 0; i < grid_depth - 1; ++i) {
            grid_weights_vector[i](position.x, position.y) += mass;
        }
        auto coords = grid_points.get_coords(position.x, position.y);
        //todo use coords
        grid_points.weight(coords.first, coords.second) += mass;
        grid_points(coords.first, coords.second).push_back(point_index);
    }
    /*void add(const Point& p, const std::size_t& point_index) {
        for (uint_fast32_t i = 0; i < grid_depth - 1; ++i) {
            grid_weights_vector[i](p.position.x, p.position.y) += p.mass;
        }
        auto coords = grid_points.get_coords(p.position.x, p.position.y);
        grid_points.weight(p.position.x, p.position.y) += p.mass;
        grid_points(p.position.x, p.position.y).push_back(point_index);
    }*/
};


class Particles : public sf::Drawable, public sf::Transformable 
{
public:
    Particles(sf::PrimitiveType pt, std::size_t size):
        vertexes(pt,size)
        ,vertex_count(size)
    {
        vertexes.resize(size);
        vertexes_copy->resize(size);
        p_position.resize(size, {{0},{0}});
        p_speed.resize(size, {{0},{0}});
        p_accel.resize(size, {{0},{0}});
        p_accel_old.resize(size, {{0},{0}});
    }

    Particles(sf::PrimitiveType pt):
        Particles(pt,0)
    {};

protected:
    std::size_t vertex_count;
    sf::VertexArray vertexes;
    //to get around stupid const
    const std::unique_ptr<sf::VertexArray> vertexes_copy{ new sf::VertexArray() };
    std::vector<Vector2d> p_position;
    std::vector<Vector2d> p_speed;
    std::vector<Vector2d> p_accel;
    std::vector<Vector2d> p_accel_old;

    const std::unique_ptr<std::mutex> position_mutex{new std::mutex()};

    virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const
    {

        for (std::size_t i = 0; i < vertex_count; ++i) {
            (*vertexes_copy)[i].position.x = p_position[i].x;
            (*vertexes_copy)[i].position.y = p_position[i].y;
        }

        states.transform *= getTransform();

        // our particles don't use a texture
        states.texture = NULL;

        // draw the vertex array
        //target.draw(*vertexes_copy, states);
        //change p_position to ptr
        target.draw(*vertexes_copy, states);
    }
private:
};

struct attractor {
    const type t;

    attractor(type t) :
        t(t) 
    {
        /*std::random_device rd{};
        auto seed = rd();
        std::cout << "Seed=" << seed<<"\n";
        gen.seed(seed);*/
    };

    void operator()(Point& p1, Point& p2, double const& re) const {
        switch (t) {
        case type::classic:
            classic_gravity_nobranch(p1, p2);
            break;
        case type::probabilistic:
            probabilistic_gravity(p1, p2, re);
            break;
        }
    };

    /*void probabilistic_gravity(Point& p1, Point& p2, double const& rand_numb)  const {
        double d = distance_squared(p1.position, p2.position);
        if (rand_numb * d < 1) {
#ifdef CLOSE_BOOST
            if (d < 1) {
                d = d < minerror ? minerror : d;
                auto sqrt_distance = std::sqrt(d);
                auto acceleration = normalize_on_second(p1.position, p2.position, sqrt_distance);
                acceleration *= gravity_constant;
                p2.accel -= acceleration * p1.mass;
                p1.accel += acceleration * p2.mass;
                return;
            }
#endif
            auto sqrt_distance = std::sqrt(d);
            auto acceleration = normalize_on_second(p1.position, p2.position, sqrt_distance);
            acceleration *= gravity_constant;
            p2.accel -= acceleration * p1.mass;
            p1.accel += acceleration * p2.mass;
        }
    }*/

    void probabilistic_gravity_first(Point& p1, Point& p2, double const& rand_numb)  const {
        double d = distance_squared(p1.position, p2.position);
        if (rand_numb * d < 1) {
#ifdef CLOSE_BOOST
            if (d < 1) {
                d = d < minerror ? minerror : d;
                auto sqrt_distance = std::sqrt(d);
                auto acceleration = normalize_on_second(p1.position, p2.position, sqrt_distance);
                acceleration *= gravity_constant;
                p2.accel -= acceleration * p1.mass;
                p1.accel += acceleration * p2.mass;
                return;
            }
#endif
            auto sqrt_distance = std::sqrt(d);
            auto acceleration = normalize_on_second(p1.position, p2.position, sqrt_distance);
            acceleration *= gravity_constant;
            p2.speed -= acceleration * p1.mass;
            p1.speed += acceleration * p2.mass;
        }
    }

    void probabilistic_gravity(Point& p1, Point& p2, double const& rand_numb)  const {
        double d = distance_squared(p1.position, p2.position);
        if (rand_numb * d < 1) {
#ifdef CLOSE_BOOST
            if (d < 1) {
                d = d < minerror ? minerror : d;
                auto sqrt_distance = std::sqrt(d);
                auto acceleration = normalize_on_second(p1.position, p2.position, sqrt_distance);
                acceleration *= gravity_constant;
                p2.accel -= acceleration * p1.mass;
                p1.accel += acceleration * p2.mass;
                return;
            }
#endif
            auto sqrt_distance = std::sqrt(d);
            auto acceleration = normalize_on_second(p1.position, p2.position, sqrt_distance);
            acceleration *= gravity_constant;
            p2.speed -= acceleration * p1.mass;
            p1.speed += acceleration * p2.mass;
        }
    }

    void probabilistic_gravity_onesided(Point& p1, Point& p2, double const& rand_numb)  const {
        double d = distance_squared_error(p1.position, p2.position);
        if (rand_numb * d < 1) {
            auto sqrt_distance = std::sqrt(d);
            auto acceleration = normalize_on_second(p1.position, p2.position, sqrt_distance);
            if (d < 1) {
                d = d < minerror ? minerror : d;
                acceleration.x /= sqrt_distance;
                acceleration.y /= sqrt_distance;
            }
            acceleration.x *= gravity_constant;
            acceleration.y *= gravity_constant;
            p1.accel.x += acceleration.x * p2.mass;
            p1.accel.y += acceleration.y * p2.mass;
        }
    }

    void classic_gravity_nobranch(Point& p1, Point& p2) const {
        Vector2d inversesquare = inverse_square_on_second_nobranch(p1.position, p2.position);
        inversesquare.x *= gravity_constant;
        inversesquare.y *= gravity_constant;
        p2.speed.x -= inversesquare.x * p1.mass;
        p2.speed.y -= inversesquare.y * p1.mass;
        p1.speed.x += inversesquare.x * p2.mass;
        p1.speed.y += inversesquare.y * p2.mass;
    }
};


class GravityParticles : public Particles {
public:
    attractor attr;

    static const uint_fast32_t threads = 2u;
    static const uint_fast32_t cache_line_length = 1u;
    static const double max_minstdrand;
    const std::vector<std::vector<std::pair<size_t, size_t>>> thread_pairs = round_robin(threads*2);
    std::minstd_rand minstd_rands[threads*2u];
    rand65536 rnd65536;
    std::size_t sections_start[threads * 2u] = {0};
    std::size_t sections_end[threads * 2u] = { 0 };
    std::thread thread_array[threads*2u];

    GravityParticles(sf::PrimitiveType pt, type t) :
        Particles(pt),
        attr(t)
    {};
    GravityParticles(sf::PrimitiveType pt, std::size_t size, type t) 
        :Particles(pt,size),
        attr(t)
    {
        std::uniform_real_distribution<double> uid(0, radius);
        std::uniform_real_distribution<double> angleud(0, 2*std::numbers::pi);
        for (std::size_t i = 0; i < vertex_count; ++i) {
            /*do {
                p_position[i] = { uid(minstd_rands[0]),uid(minstd_rands[0]) };
            } while (p_position[i].x * p_position[i].x + p_position[i].y * p_position[i].y > radius * radius);*/
            double angle = angleud(minstd_rands[0]);
            do {
                p_position[i] = uid(minstd_rands[0]) *Vector2d{ std::cos(angle),std::sin(angle) };
            } while (p_position[i].x * p_position[i].x + p_position[i].y * p_position[i].y > radius * radius);
        }
        sections_start[0] = 0;
        if (vertex_count <= cache_line_length)
            sections_end[0] = vertex_count;
        else {
            sections_end[0] = static_cast<uint_fast32_t>(vertex_count) / (threads * 2u);
            sections_end[0] = ((sections_end[0]+1) / cache_line_length) * cache_line_length;
            for (uint_fast32_t i = 1; i < threads * 2 && sections_end[i-1]< vertex_count; ++i) {
                //stupid warning even after the cast 
                sections_end[i] = ((i + uint_fast32_t{ 1 }) * static_cast<uint_fast32_t>(vertex_count)) / (threads * 2u);
                sections_end[i] = (sections_end[i]/ cache_line_length) * cache_line_length;
                sections_start[i] = sections_end[i - 1];
            }
            if (sections_end[threads * 2 - 1] > vertex_count)
                sections_end[0] = vertex_count;
        }
    }

    void perfect_setup() {
        const double x = radius*0.1;
        p_position[0] = { x,x };
        p_position[1] = { -x,-x };
    }

    void onestep(std::size_t const& i) {
        p_speed[i].x += p_accel[i].x;
        p_speed[i].y += p_accel[i].y;
        p_accel[i].x = 0;
        p_accel[i].y = 0;
        p_position[i].x += p_speed[i].x;
        p_position[i].y += p_speed[i].y;
    }
    
    void update_first() {
        for (std::size_t i = 0; i < vertex_count; ++i) {
            for (std::size_t j = i + 1; j < vertex_count; ++j) {
                Point a = Point(p_position[i], p_speed[i], p_accel[i], 1);
                Point b = Point(p_position[j], p_speed[j], p_accel[j], 1);
                attr.probabilistic_gravity(a, b, rnd65536());
            }
        }
        for (std::size_t i = 0; i < vertex_count; ++i) {
            p_speed[i] += p_accel[i]/2.;
            p_accel[i] = {};
            p_position[i] += p_speed[i];
        }
    }

    void update() {
        for (std::size_t i = 0; i < vertex_count; ++i) {
            for (std::size_t j = i + 1; j < vertex_count; ++j) {
                Point a = Point(p_position[i], p_speed[i], p_accel[i], 1);
                Point b = Point(p_position[j], p_speed[j], p_accel[j], 1);
                attr.probabilistic_gravity(a, b, rnd65536());
            }
        }
        for (std::size_t i = 0; i < vertex_count; ++i) {
            onestep(i);
        }
    }

    grid_container gc{ 16, 3, 3 };

    void init_grid() {
        gc.precalc_grid_ratio(-2000, 2000, -2000, 2000);
        //todo add random
    }

    void add_to_grid() {
        for (std::size_t i = 0; i < vertex_count; ++i) {
            gc.add(p_position[i], 1., i);
        }
    }

    template<class NormalizedRandomEngine>
    void grid_section_calc(NormalizedRandomEngine& minstdrand, std::size_t section) {
        const std::size_t a_start = sections_start[section];
        const std::size_t a_end = sections_end[section];
        for (std::size_t i = a_start; i < a_end; ++i) {

        }
    }

    template<class NormalizedRandomEngine, template<class> typename container>
    void grid_gravity_calc_topleft(NormalizedRandomEngine& minstdrand, uint_fast32_t i, uint_fast32_t j , container<std::size_t> p_indexes) {
        for (auto const& p_index : p_indexes) {
            Point p = Point(p_position[p_index], p_speed[p_index], p_accel[p_index], 1);
            //auto pos = gc.grid_points.get_coords(p_position[p_index]);
            const container<std::size_t>& subjects = gc.grid_points(i, j);
            for (const std::size_t interaction_source : gc.grid_points(i, j)) {
                //todo maybe remove if
                if (p_index == interaction_source)
                    continue;

                Point s = Point(p_position[p_index], p_speed[p_index], p_accel[p_index], 1);
            }
        }
    }

    bool time_to_join = false;
    std::condition_variable main_cv;
    std::condition_variable worker_cv;
    std::mutex main_mutex;
    std::mutex worker_m;
    uint_fast32_t threads_done = 0;
    SafeQueue<std::pair<std::size_t, std::size_t> > pair_queue;

    void wait_work(std::condition_variable& cv, std::mutex& worker_m) {
        rand65536 minstdrand2;
        do {
            std::unique_lock<std::mutex> lck(worker_m);
            cv.wait(lck, [&]() { return (pair_queue.size() > 0) || time_to_join; });
            if (time_to_join) return;
            auto p = pair_queue.pop();
            cv.notify_one();
            {
                std::scoped_lock lck(main_mutex);
                ++threads_done;
            }
            main_cv.notify_one();
        } while (!time_to_join); //data race
    }

    void start_work() {
        for (uint_fast32_t i=0; i<threads*2; ++i)
            thread_array[i] = std::thread(&GravityParticles::wait_work, this, std::ref(worker_cv), std::ref(worker_m));
    }

    void continue_work() {
        for (const auto& q : thread_pairs) {
            if (q.size() > threads) {
                std::swap(p_accel, p_accel_old);
            }
            pair_queue.push(q);
            worker_cv.notify_one();
            std::unique_lock<std::mutex> lck(main_mutex);
            main_cv.wait(lck, [&]() {return threads_done >= q.size(); });
            threads_done = 0;
        };
    }

    void stop_work() {
        {
            std::scoped_lock lck(worker_m);
            time_to_join = true;
        }
        worker_cv.notify_all();
        for (uint_fast32_t i = 0; i < threads * 2; ++i)
            if (thread_array[i].joinable()) thread_array[i].join();
    }

    void print() {
        std::cout << std::setiosflags(std::ios_base::scientific);
        std::cout << std::setprecision(3);
        for (std::size_t i = 0; i < vertex_count; ++i) {
            Point a = Point(p_position[i], p_speed[i], p_accel[i], 1);
            std::cout << "{" << i << "," << a.speed.x << "," << a.speed.y << "}";
        }
        std::cout << "\n";
    }

    void print_speed() {
        std::cout << std::setiosflags(std::ios_base::trunc);
        std::cout << std::setprecision(3);
        std::cout << "Speed[0] = " << p_speed[0].x << "," << p_speed[0].y;
    }
};


const double GravityParticles::max_minstdrand = { static_cast<double>(std::minstd_rand::max()) };
/*const std::vector<std::initializer_list<std::pair<size_t, size_t>>> thread_pairs = std::vector<std::initializer_list<std::pair<size_t, size_t>>> {
    std::initializer_list<std::pair<size_t, size_t>>{ std::pair<std::size_t,std::size_t>(0,1), std::pair<std::size_t,std::size_t>(2,3) },
    std::initializer_list<std::pair<size_t, size_t>>{ std::pair<std::size_t,std::size_t>(0,2), std::pair<std::size_t,std::size_t>(1,3) },
    std::initializer_list<std::pair<size_t, size_t>>{ std::pair<std::size_t,std::size_t>(0,3), std::pair<std::size_t,std::size_t>(1,2) },
    std::initializer_list<std::pair<size_t, size_t>>{ std::pair<std::size_t,std::size_t>(0,0), std::pair<std::size_t,std::size_t>(1,1), std::pair<std::size_t, std::size_t>(2, 2), std::pair<std::size_t, std::size_t>(3, 3) }
};*/


/*
bi tree
O(n) check all points to see how big the rectangle around 0,0 is

*/