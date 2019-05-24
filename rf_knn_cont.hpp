#ifndef CRF_KNN_H__
#define CRF_KNN_H__

#include <string>
#include <sstream>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <vector>
#include <stack>
#include <iostream>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <ctime>

#include "supervised_classifier.hpp"
#include "utils.hpp"
#include "cdt.hpp"
#include "rf_cont.hpp"

using Inst_sp = std::shared_ptr<Instance<double, std::string>>;

class KNNDoc {
 public:
  KNNDoc(Inst_sp instance, double weight) : instance_(instance), weight_(weight) {}
  KNNDoc(const KNNDoc &rhs) {
    instance_ = rhs.instance_;
    weight_ = rhs.weight_;
  }
  bool operator<(const KNNDoc &rhs) const {return instance_->id() < rhs.instance_->id();}
  bool operator==(const KNNDoc &rhs) const {return instance_->id() == rhs.instance_->id();}
  Inst_sp instance_;
  mutable double weight_;
};

namespace std {
  template <>
    class hash<KNNDoc>{
      public :
        size_t operator()(const KNNDoc &d) const {
          return hash<std::string>()(d.instance_->id());
        }
    };
};

class KNNSim {
 public:
  KNNSim(Inst_sp instance, double score) : instance_(instance), score_(score) {}
  KNNSim(const KNNSim &rhs) {
    instance_ = rhs.instance_;
    score_ = rhs.score_;
  }
  bool operator<(const KNNSim &rhs) const {return score_ < rhs.score_;}
  bool operator==(const KNNSim &rhs) const {return score_ == rhs.score_;}
  Inst_sp instance_;
  double score_;
};

class CRF_KNN : public SupervisedClassifier {
  public:
    CRF_KNN(unsigned int r, double m=1.0, unsigned int k=30, unsigned int num_trees=10)
      : SupervisedClassifier(r), num_trees_(num_trees), k_(k), m_(m) {docs_processed_ = 0; std::cerr << k << std::endl;}
    ~CRF_KNN();
    bool parse_train_line(const std::string&);
    void train(const std::string&);
    void parse_test_line(const std::string&);
    void reset_model();
    Scores<double> classify(Instance<double, std::string> &,
                            std::unordered_map<Inst_sp, double> &);
  private:
    std::vector<Inst_sp> docs_;
    double m_;
    unsigned int k_;
    unsigned int num_trees_;
    
    void insert_knn_entry(unsigned int fid, Inst_sp doc, double val);
    void updateDocSizes(); // and update doc sizes

    std::unordered_map<unsigned int, std::unordered_set<KNNDoc>> inv_idx_;
    std::map<std::string, double> knn_doc_sizes_;
    unsigned int docs_processed_;
};

void CRF_KNN::insert_knn_entry(unsigned int fid, Inst_sp doc, double val){
  KNNDoc  posting(doc, val);
  std::unordered_map<unsigned int, std::unordered_set<KNNDoc>>::iterator it = inv_idx_.find(fid);
  if (it == inv_idx_.end()) {
    std::unordered_set<KNNDoc> doc_set;
    doc_set.insert(posting);
    inv_idx_.insert(std::make_pair(fid, doc_set));
  }
  else {
    (it->second).insert(posting);
  }
}

void CRF_KNN::updateDocSizes() {
  std::unordered_map<unsigned int, std::unordered_set<KNNDoc> >::iterator it = inv_idx_.begin();
  while (it != inv_idx_.end()){
    std::unordered_set<KNNDoc>::iterator it_d = (it->second).begin();
    while(it_d != (it->second).end()){
      knn_doc_sizes_[it_d->instance_->id()] += it_d->weight_ * it_d->weight_;
      ++it_d;
    }
    ++it;
  } 
}

CRF_KNN::~CRF_KNN(){
  reset_model();
}

void CRF_KNN::reset_model(){
  docs_.clear();
  m_ = 1.0;
  num_trees_ = 0;
  k_ = 30;
}

bool CRF_KNN::parse_train_line(const std::string& line){
  std::vector<std::string> tokens; tokens.reserve(100);
  Utils::string_tokenize(line, tokens, ";");
  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return false;

  Inst_sp doc = Inst_sp(new Instance<double, std::string>());
  doc->id(tokens[0]);
  doc->output(tokens[1]);
  for (size_t i = 2; i < tokens.size()-1; i+=2) {
    unsigned int fid = std::stoi(tokens[i]);
    double val = std::stod(tokens[i+1]);
    doc->add(fid, val);
    insert_knn_entry(fid, doc, val);
  }
  docs_.push_back(doc);
  return true;

}

void CRF_KNN::train(const std::string& train_fn){
  SupervisedClassifier::train(train_fn);
  updateDocSizes();
}

Scores<double> CRF_KNN::classify(Instance<double, std::string> &doc,
                                std::unordered_map<Inst_sp, double> &sim){
  Scores<double> similarities(doc.id(), doc.output());
  std::priority_queue<KNNSim, std::vector<KNNSim> > ordered_docs;
  std::unordered_map<Inst_sp, double>::iterator it = sim.begin();
  while(it != sim.end()){
    double s = it->second;
    switch(dist_type) {
      case L2:
        s = 1.0 - sqrt(s);
        break;
      case L1:
        s = 1.0 - s;
        break;
    }
    KNNSim pqel(it->first, s);
    ordered_docs.push(pqel);
    ++it;
  }
  CRF * rf = new CRF(round, m_, num_trees_);
  rf->set_doc_delete(false);
  unsigned count = 0;
  while(!ordered_docs.empty() && count < k_){
    KNNSim pqel = ordered_docs.top();
    rf->add_document(pqel.instance_);
    ordered_docs.pop();
    count++;
  }

  rf->build();

  similarities = rf->classify(doc);
  delete rf;
  return similarities;

  /*
  unsigned count = 0;
  std::map<std::string, double> class_scores;
  while(!ordered_docs.empty() && count < k_){
    KNN_Doc_Sim pqel = ordered_docs.top();
    class_scores[pqel.doc->get_class()] += pqel.sim;
    ordered_docs.pop();
    count++;
  }
  std::map<std::string, double>::iterator it_c = class_scores.begin();
  while(it_c != class_scores.end()){
    similarities.add(it_c->first, it_c->second);
    ++it_c;
  }
  return similarities;
  */
}

void CRF_KNN::parse_test_line(const std::string& line){
  std::vector<std::string> tokens; tokens.reserve(100);
  Utils::string_tokenize(line, tokens, ";");

  double test_size = 0.0;
  std::unordered_map<Inst_sp, double> similarities;

  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return;

  Instance<double, std::string> doc;
  doc.id(tokens[0]);
  doc.output(tokens[1]);

  for (size_t i = 2; i < tokens.size()-1; i+=2) {
    unsigned int fid = std::stoi(tokens[i]);
    double val = std::stod(tokens[i+1]);
    doc.add(fid, val);
    test_size += val * val;
  }

  for (auto f : doc.bow()) {
    std::unordered_map<unsigned int, std::unordered_set<KNNDoc>>::iterator it =
      inv_idx_.find(f.first);
    if (it != inv_idx_.end()) {
      std::unordered_set<KNNDoc>::iterator it_s = (it->second).begin();
      while(it_s != (it->second).end()) {
        double val = f.second;
        switch(dist_type) {
          case L2:
            similarities[it_s->instance_] += pow((it_s->weight_/sqrt(knn_doc_sizes_[it_s->instance_->id()])) - (val/sqrt(test_size)), 2.0);
            break;
          case L1:
            similarities[it_s->instance_] += abs((it_s->weight_/sqrt(knn_doc_sizes_[it_s->instance_->id()])) - (val/sqrt(test_size)));
            break;
          case COSINE:
          default:
            similarities[it_s->instance_] += (it_s->weight_/sqrt(knn_doc_sizes_[it_s->instance_->id()])) * (val/sqrt(test_size));
            break;
        }
        ++it_s;
      }
    }
  }

  Scores<double> scores = classify(doc, similarities);
  docs_processed_++;
  std::cerr.precision(4);
  std::cerr.setf(std::ios::fixed);
  std::cerr << "\r" << double(docs_processed_)/docs_.size() * 900 << "%";
  
  get_outputer()->output(scores);
}

#endif
