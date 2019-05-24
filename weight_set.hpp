#ifndef WEIGHT_SET_HPP__
#define WEIGHT_SET_HPP__

#include <map>

template<typename T>
class WeightSet {
 public:
  WeightSet() {}

  double get(T i) {
    if (w_.find(i) == w_.end()) return 1.0;
    return w_[i];
  }

  double set(T i, double w) {
    return w_[i] = w;
  }

  bool empty() { return w_.size(); }

 private:
  std::map<T, double> w_;

};

#endif
