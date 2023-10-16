#pragma once
#include<queue>
#include<deque>
#include<list>
#include<map>
#include<mutex>
#include<optional>

template<class Event>
class SafeQueue {
protected:
    std::queue<Event> q;
    std::mutex m;
public:
    std::queue<Event>& data() {
        return q;
    }
    template<typename Function, typename... Args>
    void do_locking_operation(Function&& function, Args&... args) {
        std::scoped_lock<std::mutex> lock(m);
        function(q, args...);
    }
    template<typename Function, typename... Args>
    decltype(q)&& return_locking_operation(Function&& function, Args&... args) {
        std::scoped_lock<std::mutex> lock(m);
        return function(q, args...);
    }
    SafeQueue() :
        q()
    {}
    std::size_t size() {
        std::scoped_lock<std::mutex> lock(m);
        return q.size();
    }
    std::optional<Event> pop_if_not_empty() {
        std::scoped_lock<std::mutex> lock(m);
        if (q.size() > 0) {
            //T front = q.front();
            Event front = std::move(q.front());
            q.pop();
            return std::optional<Event>{front};
        }
        return {};
    }
    Event pop() {
        std::scoped_lock<std::mutex> lock(m);
        //T front = q.front();
        Event front = std::move(q.front());
        q.pop();
        return front;
    }
    void push(Event element) {
        std::scoped_lock<std::mutex> lock(m);
        q.push(element);
    }
    void push(std::initializer_list<Event> list) {
        std::scoped_lock<std::mutex> lock(m);
        for (auto l : list) {
            q.push(l);
        }
    }
    void redefine(std::queue<Event> new_q) {
        std::scoped_lock<std::mutex> lock(m);
        q = new_q;
    }
    void push(std::vector<Event> new_q) {
        std::scoped_lock<std::mutex> lock(m);
        for (auto e : new_q)
            q.push(e);
    }
};

template<class Event>
class SafeDeque {
protected:
    std::deque<Event> q;
    std::mutex m;
public:
    template<typename Function, typename... Args>
    void do_locking_operation(Function&& function, Args&... args) {
        std::scoped_lock<std::mutex> lock(m);
        function(q, args...);
    }
    //is there a way to group this with the previous function?
    template<typename Function, typename... Args>
    decltype(q) return_locking_operation(Function && function, Args&... args) {
        std::scoped_lock<std::mutex> lock(m);
        return function(q, args...);
    }
    SafeDeque() :
        q()
    {}
    std::size_t size() {
        std::scoped_lock<std::mutex> lock(m);
        return q.size();
    }
    std::optional<Event> pop_if_not_empty() {
        std::scoped_lock<std::mutex> lock(m);
        if (q.size() > 0) {
            //T front = q.front();
            Event front = std::move(q.front());
            q.pop();
            return std::optional<Event>{front};
        }
        return {};
    }
    Event pop() {
        std::scoped_lock<std::mutex> lock(m);
        //T front = q.front();
        Event front = std::move(q.front());
        q.pop_front();
        return front;
    }
    void push(Event element) {
        std::scoped_lock<std::mutex> lock(m);
        q.push_back(element);
    }
    void push(std::initializer_list<Event> list) {
        std::unique_lock<std::mutex> lock(m);
        for (auto l : list) {
            q.push(l);
        }
    }
};

template<class Event>
struct SafeList {
    //never finished this
private :
    SafeList();
protected:
    std::list<Event> l;
    std::mutex m;
public:
    template<typename Function, typename... Args>
    void do_locking_operation(Function&& function, Args&... args) {
        std::scoped_lock<std::mutex> lock(m);
        function(l, args...);
    }
    Event& front() {
        std::scoped_lock<std::mutex> lock(m);
        return l.front();
    }
    const Event& front() const {
        std::scoped_lock<std::mutex> lock(m);
        return l.front();
    }
    Event& back() {
        std::scoped_lock<std::mutex> lock(m);
        return l.back();
    }
    const Event& back() const {
        std::scoped_lock<std::mutex> lock(m);
        return l.back();
    }
};


template<typename key, class Event> 
struct SafeMap {
private:
    SafeMap();
protected:
    std::map<key, Event> map;

};