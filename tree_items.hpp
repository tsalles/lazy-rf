#ifndef TREE_ITEMS_HPP__
#define TREE_ITEMS_HPP__

#include <memory>
#include <unordered_set>
#include <vector>

#include <cmath>

#include "tree_instance.hpp"

template<typename FTYPE, typename OTYPE>
class Items {

 public:

  Items() {}

  Items(const Items<FTYPE, OTYPE> &rhs) {
    items_.insert(items_.begin(), rhs.items_.begin(), rhs.items_.end());
    f_set_.insert(rhs.f_set_.begin(), rhs.f_set_.end());
    class_dist_.insert(rhs.class_dist_.begin(), rhs.class_dist_.end());
  }

  std::shared_ptr<Instance<FTYPE, OTYPE>> get(unsigned int i) { return items_[i]; }

  void add (std::shared_ptr<Instance<FTYPE, OTYPE>> i) {
    items_.push_back(i);
    class_dist_[i->output()]++;
    for (const auto e : i->bow()) {
      f_set_.insert(e.first);
    }
  }

  std::vector<std::shared_ptr<Instance<FTYPE, OTYPE>>> &items() { return items_; }

  std::unordered_set<unsigned int> &features() { return f_set_; }

  const std::unordered_map<OTYPE, unsigned int> &class_dist() const { return class_dist_; }

  double entropy() {
    double e = 0;
    for (const auto c : class_dist_) {
      double p = static_cast<double>(c.second)/static_cast<double>(items_.size());
      e += p * log2(p);
    }
    return -e;
  }

  unsigned int size() { return items_.size(); }

  void clear() {
    items_.clear();
    f_set_.clear();
    class_dist_.clear();
  }

 private:
  std::vector<std::shared_ptr<Instance<FTYPE, OTYPE>>> items_;
  std::unordered_set<unsigned int> f_set_;
  std::unordered_map<OTYPE, unsigned int> class_dist_;

};

#endif
