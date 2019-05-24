#ifndef CDT_HPP__
#define CDT_HPP__

#include <algorithm>
#include <tuple>
#include <unordered_map>

#include "supervised_classifier.hpp"
#include "tree_items.hpp"

class CDT : public virtual SupervisedClassifier {

 public:
  CDT(unsigned int r=0, unsigned int maxh=0, double m=1.0) : SupervisedClassifier(r),
    left_(nullptr), right_(nullptr), level_(0), cut_feature_(0), cut_value_(double()), maxh_(maxh), m_(m) {}
  CDT(const Items<double, std::string> &items, unsigned int r=0, unsigned int maxh=0, double m=1.0)
    : SupervisedClassifier(r), left_(nullptr), right_(nullptr), items_(items),
      level_(0), cut_feature_(0), cut_value_(double()), maxh_(maxh), m_(m) {}

  ~CDT() {
    if (left_  != nullptr) delete left_;
    if (right_ != nullptr) delete right_;
  }

  virtual bool parse_train_line(const std::string &line) {
    std::vector<std::string> tks; tks.reserve(100);
    Utils::string_tokenize(line, tks, ";");
    // input format: doc_id;class_name;{term_id;tf}+
    if ((tks.size() < 4) || (tks.size() % 2 != 0)) return false;
  
    std::shared_ptr<Instance<double, std::string>> instance =
      std::shared_ptr<Instance<double, std::string>>(new Instance<double, std::string>());
    instance->output(tks[1]);
    for (size_t i = 2; i < tks.size()-1; i+=2) {
      unsigned int fid = std::stoi(tks[i]);
      double val = tks[i+1] == "?" ? 0.0 : std::stod(tks[i+1]);
      instance->add(fid, val);
    }
    items_.add(instance);
 
    return true;
  }

  virtual void parse_test_line(const std::string &line) {
    std::vector<std::string> tks; tks.reserve(100);
    Utils::string_tokenize(line, tks, ";");
    if ((tks.size() < 4) || (tks.size() % 2 != 0)) return;

    Instance<double, std::string> instance;
    instance.output(tks[1]);
    for (size_t i = 2; i < tks.size()-1; i+=2) {
      unsigned int fid = std::stoi(tks[i]);
      double val = tks[i+1] == "?" ? 0.0 : std::stod(tks[i+1]);
      instance.add(fid, val);
    }

    auto pred = classify(instance);
    Scores<double> similarities(tks[0], tks[1]);
    double normalizer = 0.0;
    for (const auto p : pred) {
      similarities.add(p.first, static_cast<double>(p.second));
      normalizer += p.second;
    }

    get_outputer()->output(similarities, normalizer);
  }

  virtual bool check_train_line(const std::string &line) const {
    std::vector<std::string> tokens; tokens.reserve(100);
    Utils::string_tokenize(line, tokens, ";");
    // input format: doc_id;class_id;{term_id;tf}+
    if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return false;
    return true;
  }

  virtual void reset_model() {
    left_ = nullptr;
    right_ = nullptr;
    items_.clear();
    level_ = 0;
    cut_feature_ = 0;
    cut_value_ = double();
  }

  virtual void train(const std::string &trn) {
    SupervisedClassifier::train(trn);
    build();
  }

  std::unordered_map<std::string, unsigned int> classify(Instance<double, std::string> &instance) {
    if (!stop()) {
      if (instance.get(cut_feature_) < cut_value_) {
        if (left_ != nullptr) {
          return left_->classify(instance);
        }
      }
      else {
        if (right_ != nullptr) {
          return right_->classify(instance);
        }
      }
    }

    return items_.class_dist();
  }

  void build() {
    if (!stop()) {
      auto split = find_split();
      cut_feature_ = std::get<1>(split);
      cut_value_   = std::get<2>(split);
      Items<double, std::string> left  = std::get<3>(split);
      Items<double, std::string> right = std::get<4>(split);
      if (left.size() > 0) {
        left_ = new CDT(left, round, maxh_, m_);
        left_->level(level_+1);
        left_->build();
      }
      if (right.size() > 0) {
        right_ = new CDT(right, round, maxh_, m_);
        right_->level(level_+1);
        right_->build();
      }
      //items_.clear();
    }
  }

 private:
  CDT *left_;
  CDT *right_;

  Items<double, std::string> items_;
  unsigned int level_;
  unsigned int cut_feature_;
  double cut_value_;
  unsigned int maxh_;
  double m_;

  bool stop() {
    return (items_.size() <= 1 || items_.class_dist().size() == 1);
  }

  void level(unsigned int lvl) { level_ = lvl; }

  std::tuple<double, unsigned int, double,
             Items<double,std::string>,
             Items<double,std::string>> find_split() {

    Items<double, std::string> /*left, right,*/ bst_left, bst_right;

    double min_entropy = 999999.99;
    unsigned int cut_feature = 0; // FIXME
    double cut_value = double();

    unsigned int m_int = ceil(m_ * items_.features().size()); // original feature set
    std::vector<unsigned int> selected;
    selected.reserve(m_int);
    selected.insert(selected.begin(), items_.features().begin(), items_.features().end());
    if (m_int < items_.features().size()) {
      std::random_shuffle(selected.begin(), selected.end());
      selected.resize(m_int);
    }

    bool found = false;

    #pragma omp parallel for
    for (unsigned int sel_idx = 0; sel_idx < selected.size(); sel_idx++) {
      #pragma omp flush(found)
      if (!found) {
        std::vector<std::shared_ptr<Instance<double, std::string>>> items(items_.items());

        unsigned int feature = selected[sel_idx];
        Items<double, std::string> left, right;
        //left.clear(); 
        // sort items by feature value
        std::sort(items.begin(), items.end(),
          [feature] (std::shared_ptr<Instance<double, std::string>> a,
              std::shared_ptr<Instance<double, std::string>> b) {
            return a->get(feature) < b->get(feature);   
        });
        
        // incrementally build left and right nodes, computing entropy and deciding best split
        auto linst = items[0];
        left.add(linst);
        if (items.size() > 1) {
          for (unsigned int i = 1; i < items.size()-1; i++) {
            auto rinst = items[i];
            if (linst->output() == rinst->output()) {
              left.add(rinst);
            }
            else {
              right.add(rinst);
              for (unsigned int j = i+1; j < items.size(); j++) {
                right.add(items[j]);
              }
              double lent = left.entropy();
              double rent = right.entropy();
              double avgent = (lent + rent) / 2.0;
              #pragma omp critical(best_split_update)
              {
                if (avgent < min_entropy) {
                  min_entropy = avgent;
                  cut_feature = feature;
                  cut_value = rinst->get(feature);
                  bst_left = left;
                  bst_right = right;
                  if (min_entropy < 1.0e-5) {
                    found = true;
                    //return std::make_tuple(min_entropy, cut_feature, cut_value, bst_left, bst_right);
                  }
                }

                if (!found) {
                  left.add(rinst);
                  right.clear();
                }
              }
            }
          }
        }

        if (left.size() <= 1) { // singleton or pure node.
          #pragma omp critical(best_split_update)
          {
            min_entropy = 0;
            cut_feature = feature;
            cut_value = linst->get(feature);
            bst_left = left;
            bst_right = right;
            found = true;
          }
          //return std::make_tuple(min_entropy, cut_feature, cut_value, bst_left, bst_right);
        }
      }
    }

    return std::make_tuple(min_entropy, cut_feature, cut_value, bst_left, bst_right);
  }


};

#endif
