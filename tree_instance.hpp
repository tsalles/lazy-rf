#ifndef TREE_INSTANCE_HPP__
#define TREE_INSTANCE_HPP__

#include <unordered_map>

template<typename FTYPE, typename OTYPE>
class Instance {

 public:
  Instance() {}

  Instance(std::unordered_map<unsigned int, FTYPE> &bow, OTYPE output)
    : bow_(bow), output_(output) {}

  const std::unordered_map<unsigned int, FTYPE> &bow() const { return bow_; }
  OTYPE output() const { return output_; }
  void output(OTYPE output) { output_ = output; }
  std::string id() { return id_; }
  void id(std::string id) { id_ = id; }

  void add(const unsigned int id, const FTYPE val) { bow_[id] = val; }
  FTYPE get(const unsigned int id) const {
    auto it = bow_.find(id);
    if (it != bow_.end()) return it->second;
    return FTYPE();
  }

 private:
  std::unordered_map<unsigned int, FTYPE> bow_;
  OTYPE output_;
  std::string id_;

};

#endif

