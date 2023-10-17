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
#include<type_traits>
#include"myrand.hpp"

typedef sf::Vector2<double> Vector2d; 
template<class T>
using container_type = std::vector<T>;
using namespace std::literals;

struct Point {
    Vector2d& position;
    Vector2d& speed;
    //Vector2d& accel;
    const double mass{ 1 };
    Point(Vector2d& position, Vector2d& speed, double mass)
        :position(position), speed(speed), mass(mass) {};
    //Point(Vector2d& position, Vector2d& speed, Vector2d& accel, double mass)
    //    :position(position), speed(speed), accel(accel), mass(mass) {};
};

struct GravitySource {
    Vector2d const& position;
    const double& mass;
    GravitySource(Vector2d const& position, double const& mass)
        :position(position), mass(mass) {};
};

enum class type {
    classic,
    probabilistic
};


class Particles : public sf::Drawable, public sf::Transformable
{
public:
    Particles(sf::PrimitiveType pt, std::size_t size) :
        vertexes(pt, size)
        , vertex_count(size)
    {
        vertexes.resize(size);
        vertexes_copy->resize(size);
        p_position.resize(size, { {0},{0} });
        p_speed.resize(size, { {0},{0} });
    }

    Particles(sf::PrimitiveType pt) :
        Particles(pt, 0)
    {};

protected:
    std::size_t vertex_count;
    sf::VertexArray vertexes;
    //to get around stupid const
    const std::unique_ptr<sf::VertexArray> vertexes_copy{ new sf::VertexArray() };
    std::vector<Vector2d> p_position;
    std::vector<Vector2d> p_speed;

    const std::unique_ptr<std::mutex> position_mutex{ new std::mutex() };

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

    void probabilistic_gravity_onesided(Point& p1, GravitySource const& p2, double const distance_squared, double const& rand_numb)  const {
        if (rand_numb * distance_squared < 1) {
            auto sqrt_distance = std::sqrt(distance_squared);
            auto acceleration = normalize_on_second(p1.position, p2.position, sqrt_distance);
            acceleration.x *= probabilistic_gravity_constant;
            acceleration.y *= probabilistic_gravity_constant;
            p1.speed.x += acceleration.x * p2.mass;
            p1.speed.y += acceleration.y * p2.mass;
        }
    }

    /*template<class UniformRandomBitGenerator>
    void probabilistic_gravity_multiple(Point& p1, GravitySource const& p2, double const& dist_squared_error, UniformRandomBitGenerator& re) const { */
    void probabilistic_gravity_multiple(Point& p1, GravitySource const& p2, int_fast32_t const& weight, double const& dist_squared_error, std::minstd_rand& re) const {
        std::binomial_distribution<> b{weight, 1. / dist_squared_error };
        double multiplier = b(re);
        auto sqrt_distance = std::sqrt(dist_squared_error);
        auto acceleration = normalize_on_second(p1.position, p2.position, sqrt_distance);
        acceleration.x *= probabilistic_gravity_constant;
        acceleration.y *= probabilistic_gravity_constant;
        p1.speed.x += acceleration.x * multiplier;
        p1.speed.y += acceleration.y * multiplier;
    }

    void classic_gravity_onesided(Point& p1, GravitySource const& p2, double const& dist_squared_error) const {
        Vector2d inversesquare = inverse_square_on_second_nobranch(p1.position, p2.position, dist_squared_error);
        inversesquare.x *= classic_gravity_constant;
        inversesquare.y *= classic_gravity_constant;
        p1.speed.x += inversesquare.x * p2.mass;
        p1.speed.y += inversesquare.y * p2.mass;
    }
};

struct quadtree_node { //add 4 of these at a time
    double weight{ 0 };
    int_fast32_t weight_int{ 0 };
    Vector2d center;
    Vector2d size;
    Vector2d center_of_mass{ 0,0 };

    //points to first child if this node is a branch (i.e. father of more nodes)
    //point to node_information 
    int_fast32_t first_child = -1; //NW NE SW NE, -1 if no children
    int_fast32_t point_index = -1; //index of the point here//maybe useless//maybe not cause it acts as a boolean
};

struct quadtree_node_plusentials {
    double weight{ 0 };
    Vector2d center;
    Vector2d size;
    int_fast32_t point_index = -1; //index of the point here//maybe useless//maybe not cause it acts as a boolean
};

struct quadtree_node_essentials { //add 4 of these at a time
    float weight{ 0 };
    float size;
    sf::Vector2f center_of_mass{ 0,0 };

    //points to first child if this node is a branch (i.e. father of more nodes)
    //point to node_information 
    int_fast32_t first_child = -1; //NW NE SW NE, -1 if no children
};

struct quadtree {
    std::vector<quadtree_node> nodes;
    std::vector<quadtree_node_essentials> nodes_essentials;
    std::vector<quadtree_node_plusentials> nodes_plusentials;
    std::vector<Vector2d>& p_position;
    Vector2d topleft;
    Vector2d bottomright;
    std::size_t reserv;

    quadtree(std::vector<Vector2d>& p_position, Vector2d const& topleft, Vector2d const& bottomright, std::size_t reserv=nodes_reserved) :
        p_position(p_position),
        topleft(topleft),
        bottomright(bottomright),
        reserv(reserv)
    {
        nodes.reserve(reserv);
        reset();
    };

    void reset() {
        nodes.clear();
        nodes.push_back(quadtree_node{ .weight = 0, .center = (topleft + bottomright) / 2., .size = bottomright - topleft });
        nodes_essentials.clear();
        nodes_essentials.push_back(quadtree_node_essentials{ .weight = 0 });
        nodes_plusentials.clear();
        nodes_plusentials.push_back(quadtree_node_plusentials{ .center = (topleft + bottomright) / 2., .size = bottomright - topleft });
    }

    void reset(Vector2d const& topleft, Vector2d const& bottomright) {
        this->topleft = topleft;
        this->bottomright = bottomright;
        reset();
    }

    void add_point(std::size_t point_index, int_fast32_t node_index=0) {
        //dfs
        uint_fast32_t depth = 0;
        quadtree_node& node = nodes[node_index];
        /*auto& nodee = nodes_essentials[node_index];
        auto& nodep = nodes_plusentials[node_index];*/
        if (node.first_child==-1 && node.point_index == -1) { //empty branch
            node.point_index = point_index;
            //nodep.point_index = point_index;
            node.center_of_mass = p_position[point_index];
            //nodee.center_of_mass.x = p_position[point_index].x;
            //nodee.center_of_mass.y = p_position[point_index].y;
            ++node.weight;
            //++nodee.weight;
            ++node.weight_int;
            //++nodee.weight_int;
        }
        else if (node.first_child == -1 && node.point_index != -1) { // only one node to relocate
            std::size_t old_point_index = static_cast<std::size_t>(node.point_index);
            node.point_index = -1;
            node.point_index = -1;
            int_fast32_t new_node_index = node.first_child = nodes.size();
            Vector2d center_old = node.center;
            Vector2d size_new = node.size/2.;
            Vector2d center_new = center_old- node.size / 4.;
            Vector2d topleft_old = node.center - size_new;
            Vector2d bottomright_old = node.center + size_new;

            //update center of mass //Todo if different weights
            //gotta do it before push_back or iterators are invalidated
            node.center_of_mass = (p_position[point_index] * 1. + node.center_of_mass * node.weight) / (1. + node.weight);
            ++node.weight;
            ++node.weight_int;

            nodes.push_back(quadtree_node{.center = center_new, .size = size_new });
            center_new.x += size_new.x;
            nodes.push_back(quadtree_node{.center = center_new, .size = size_new });
            center_new.x -= size_new.x;
            center_new.y += size_new.y;
            nodes.push_back(quadtree_node{.center = center_new, .size = size_new });
            center_new.x += size_new.x;
            nodes.push_back(quadtree_node{.center = center_new, .size = size_new });

            int_fast32_t index_add = 0;
            index_add += (p_position[old_point_index].x - topleft_old.x) / size_new.x;
            index_add += 2*static_cast<std::size_t>((p_position[old_point_index].y - topleft_old.y) / size_new.y);

            
            //put move_point in its leaf with add_point(move_point)
            add_point(old_point_index, new_node_index + index_add);
            //put 
            index_add = (p_position[point_index].x - topleft_old.x) / size_new.x;
            index_add += 2 * static_cast<std::size_t>((p_position[point_index].y - topleft_old.y) / size_new.y);
            add_point(point_index, new_node_index + index_add);
        }
        else {
            //update center of mass //Todo if different weights
            node.center_of_mass = (p_position[point_index] * 1. + node.center_of_mass * node.weight) / (1. + node.weight);
            ++node.weight;
            ++node.weight_int;

            Vector2d size_new = node.size / 2.;
            Vector2d topleft_old = node.center - size_new;
            Vector2d bottomright_old = node.center + size_new;
            int_fast32_t index_add = (p_position[point_index].x - topleft_old.x) / size_new.x;
            index_add += 2 * static_cast<std::size_t>((p_position[point_index].y - topleft_old.y) / size_new.y);
            add_point(point_index, node.first_child + index_add);
        }
    }
};

class GravityParticles : public Particles {
public:
    attractor attr;
    static const double max_minstdrand;
    std::minstd_rand minstd_rand;
    rand65536 rnd65536;
    std::size_t sections_start[section_count] = { 0 };
    std::size_t sections_end[section_count] = { 0 };
    std::thread thread_array[threads];
    //todo
    Vector2d topleft{ -1,-1 };
    Vector2d bottomright{ 1,1 };
    quadtree qtree{ p_position,Vector2d{-800,-800},Vector2d{800,800} };


    GravityParticles(sf::PrimitiveType pt, type t) :
        Particles(pt),
        attr(t)
    {};
    GravityParticles(sf::PrimitiveType pt, std::size_t size, type t)
        :Particles(pt, size),
        attr(t)
    {
        std::uniform_real_distribution<double> uid(0, radius);
        std::uniform_real_distribution<double> angleud(0, 2 * std::numbers::pi);
        for (std::size_t i = 0; i < vertex_count; ++i) {
            /*do {
                p_position[i] = { uid(minstd_rands[0]),uid(minstd_rands[0]) };
            } while (p_position[i].x * p_position[i].x + p_position[i].y * p_position[i].y > radius * radius);*/
            double angle = angleud(minstd_rand);
            double speed = uid(minstd_rand);
            do {
                p_position[i] = uid(minstd_rand) * Vector2d { std::cos(angle), std::sin(angle) };
            } while (p_position[i].x * p_position[i].x + p_position[i].y * p_position[i].y > radius * radius);
            /*for (int i = 0; i < vertex_count; ++i) {
                p_speed[i].x = p_position[i].y / std::sqrt(p_position[i].y* p_position[i].y+ p_position[i].x* p_position[i].x) *classic_gravity_constant*std::sqrt(radius);
                p_speed[i].y = -p_position[i].x / std::sqrt(p_position[i].y * p_position[i].y + p_position[i].x * p_position[i].x) * classic_gravity_constant * std::sqrt(radius);
            }*/
        }
        sections_start[0] = 0;
        if (vertex_count <= min_section_length)
            sections_end[0] = vertex_count;
        else {
            sections_end[0] = static_cast<uint_fast32_t>(vertex_count) / section_count;
            sections_end[0] = ((sections_end[0] + 1) / min_section_length) * min_section_length;
            for (uint_fast32_t i = 1; i < section_count && sections_end[i - 1] < vertex_count; ++i) {
                //stupid warning even after the cast 
                sections_end[i] = ((i + uint_fast32_t{ 1 }) * static_cast<uint_fast32_t>(vertex_count)) / section_count;
                sections_end[i] = (sections_end[i] / min_section_length) * min_section_length;
                sections_start[i] = sections_end[i - 1];
            }
            if (sections_end[section_count - 1] > vertex_count)
                sections_end[0] = vertex_count;
        }
    }

    double theta = 1;

    void force_on_point(Point& p, std::size_t const& point_index, int_fast32_t node_index = 0) {
        quadtree_node& node = qtree.nodes[node_index];
        if (node.weight == 0 || node.point_index == point_index)
            return;
        double dist_squared = distance_squared_error(node.center_of_mass, p.position);
        double width = node.size.x;
        if (width * width < theta * theta * dist_squared) {
            //approx
            Vector2d speed{ 0,0 };
            GravitySource papprox{ node.center_of_mass, node.weight };
            attr.classic_gravity_onesided(p, papprox, dist_squared);
            //attr.probabilistic_gravity_multiple(p, papprox, node.weight_int, dist_squared, minstd_rand); too slow
        }
        else if (node.first_child == -1) {//far enough or no children so it's always "approx" at this point
            Vector2d speed{ 0,0 };
            GravitySource papprox{ node.center_of_mass, node.weight };
            attr.probabilistic_gravity_onesided(p, papprox, dist_squared, rnd65536());
        }
        else {
            for (int_fast32_t i = 0; i < 4; ++i)
                force_on_point(p, point_index, node.first_child + i);
        }
        return;
    }

    void force_on_point(rand65536& rnd, Point& p, std::size_t const& point_index, int_fast32_t node_index = 0) {
        quadtree_node& node = qtree.nodes[node_index];
        if (node.weight == 0 || node.point_index == point_index)
            return;
        double dist_squared = distance_squared_error(node.center_of_mass, p.position);
        double width = node.size.x;
        if (width * width < theta * theta * dist_squared) {
            //approx
            Vector2d speed{ 0,0 };
            GravitySource papprox{ node.center_of_mass, node.weight };
            attr.classic_gravity_onesided(p, papprox, dist_squared);
            //attr.probabilistic_gravity_multiple(p, papprox, node.weight_int, dist_squared, minstd_rand); too slow
        }
        else if (node.first_child == -1) {//far enough or no children so it's always "approx" at this point
            Vector2d speed{ 0,0 };
            GravitySource papprox{ node.center_of_mass, node.weight };
            attr.probabilistic_gravity_onesided(p, papprox, dist_squared, rnd());
        }
        else {
            for (int_fast32_t i = 0; i < 4; ++i)
                force_on_point(rnd, p, point_index, node.first_child + i);
        }
        return;
    }

    void perfect_setup() {
        const double x = radius * 0.1;
        p_position[0] = { x,x };
        p_position[1] = { -x,-x };
    }

    void onestep(std::size_t const& i) {
        p_position[i].x += p_speed[i].x;
        p_position[i].y += p_speed[i].y;
    }

    void fill_quadtree(Vector2d const& topleft, Vector2d const& bottomright) {
        qtree.reset(topleft, bottomright);
        for (std::size_t i = 0; i < vertex_count; ++i) {
            qtree.add_point(i);
        }
    }

    void build_tree() {
        for (std::size_t i = 0; i < vertex_count; ++i) {
            topleft.x = std::min(topleft.x, p_position[i].x);
            topleft.y = std::min(topleft.y, p_position[i].y);
            bottomright.x = std::max(bottomright.x, p_position[i].x);
            bottomright.y = std::max(bottomright.y, p_position[i].y);
        }
        topleft -= {1., 1.};
        bottomright += {1., 1.};
        fill_quadtree(topleft, bottomright);
    }

    void update_tree(bool first = false, std::size_t section_start = { 0 }, std::size_t section_end = { section_count - 1 }) {
        build_tree();
        for (std::size_t i = sections_start[section_start]; i < sections_end[section_end]; ++i) {
            Point p = Point(p_position[i], p_speed[i], 1);
            force_on_point(p, i, 0);
        };
        if (first) {
            for (std::size_t i = sections_start[section_start]; i < sections_end[section_end]; ++i) {
                p_speed[i] /= 2.;
            }
        }
        for (std::size_t i = sections_start[section_start]; i < sections_end[section_end]; ++i) {
            onestep(i);
        }
    }
    void update_tree_multithreading(rand65536 & rnd, std::size_t section, bool first = false) {
        for (std::size_t i = sections_start[section]; i < sections_end[section]; ++i) {
        //for (std::size_t i = 0; i < vertex_count; ++i) {
            Point p = Point(p_position[i], p_speed[i], 1);
            force_on_point(rnd, p, i, 0);
        }
        if (first) {
            for (std::size_t i = sections_start[section]; i < sections_end[section]; ++i) {
                p_speed[i] /= 2.;
            }
        }
        for (std::size_t i = sections_start[section]; i < sections_end[section]; ++i) {
            onestep(i);
        }
    }

    bool time_to_join = false;
    std::condition_variable main_cv;
    std::condition_variable worker_cv;
    std::mutex main_mutex;
    std::mutex worker_m;
    uint_fast32_t threads_done = 0;
    uint_fast32_t threads_woken = 0;
    SafeQueue<std::size_t> section_queue;

    void wait_work(std::condition_variable& cv, std::mutex& worker_m, std::size_t section) {
        rand65536 fast_random_double;
        do {
            std::unique_lock<std::mutex> lck(worker_m);
            cv.wait(lck, [&]() { return time_to_join || (threads_woken < threads && threads_done < threads); });
            ++threads_woken;
            if (time_to_join) return; //data race
            update_tree_multithreading(fast_random_double, section, false);
            {
                std::scoped_lock lck(main_mutex);
                ++threads_done;
            }
            main_cv.notify_one();
        } while (!time_to_join); //data race
    }

    void start_work() {
        for (uint_fast32_t i = 0; i < threads; ++i)
            thread_array[i] = std::thread(&GravityParticles::wait_work, this, std::ref(worker_cv), std::ref(worker_m), i);
    }

    void continue_work() {
        build_tree(); 
        threads_woken=0;
        threads_done = 0;
        worker_cv.notify_all();
        std::unique_lock<std::mutex> lck(main_mutex);
        main_cv.wait(lck, [&]() {return threads_done >= threads; });
    }

    void stop_work() {
        {
            std::scoped_lock lck(worker_m);
            time_to_join = true;
        }
        worker_cv.notify_all();
        for (uint_fast32_t i = 0; i < threads; ++i)
            if (thread_array[i].joinable()) thread_array[i].join();
    }

    void print_speed() {
        std::cout << std::setiosflags(std::ios_base::trunc);
        std::cout << std::setprecision(3);
        std::cout << "Speed[0] = " << p_speed[0].x << "," << p_speed[0].y;
    }
};


const double GravityParticles::max_minstdrand = { static_cast<double>(std::minstd_rand::max()) };


/*
bi tree
O(n) check all points to see how big the rectangle around 0,0 is

*/