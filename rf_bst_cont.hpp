#ifndef CRF_BOOST_H__
#define CRF_BOOST_H__

#include <string>
#include <sstream>
#include <map>
#include <set>
#include <list>
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

class CRF_BOOST : public SupervisedClassifier {
  using Inst_sp = std::shared_ptr<Instance<double, std::string>>;

  public:
    CRF_BOOST(unsigned int r, double m=1.0, unsigned int max_trees=200, unsigned int maxh=0, bool trn_err=false)
      : SupervisedClassifier(r), m_(m), max_trees_(max_trees) {
       docs_processed_ = 0;
      for (unsigned int i = 0; i < max_trees_; i++) {
        CRF * crf = new CRF(round, m_, 50/*i*/, maxh, trn_err);
        crf->set_doc_delete(false);
        ensemble_[i] = crf;
      }
    }
    ~CRF_BOOST();
    bool parse_train_line(const std::string&);
    void train(const std::string&);
    void parse_test_line(const std::string&);
    void reset_model();
    Scores<double> classify(Instance<double, std::string> &,
                   std::unordered_map<Inst_sp, double>&);
  private:
    double m_;
    unsigned int max_trees_;
    unsigned int docs_processed_;
    std::unordered_map<unsigned int, CRF*> ensemble_;
};

CRF_BOOST::~CRF_BOOST() {
  reset_model();
}

void CRF_BOOST::reset_model() {
  for(int i = 0; i < ensemble_.size(); i++){
    delete ensemble_[i];
  }
}

bool CRF_BOOST::parse_train_line(const std::string& line) {
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
  }
  for (unsigned int i = 0; i < max_trees_; i++) {
    ensemble_[i]->add_document(doc);
  }
  return true;
}

void CRF_BOOST::train(const std::string& train_fn) {
  SupervisedClassifier::train(train_fn);
  WeightSet<std::string> w;
  for (unsigned int i = 0; i < max_trees_; i++) {
    std::cerr << "\rBuilding RF " << i;
    ensemble_[i]->build(&w);
  }
  std::cerr << std::endl;
}

Scores<double> CRF_BOOST::classify(Instance<double, std::string> &doc,
                                   std::unordered_map<Inst_sp, double> &sim) {
  Scores<double> similarities(doc.id(), doc.output());
  std::unordered_map<std::string, double> sco;
  #pragma omp parallel for schedule(static)
  for (unsigned int i = 0; i < max_trees_; i++) {
    double oob_err = ensemble_[i]->avg_oob_err();
    Scores<double> s = ensemble_[i]->classify(doc);
    while (!s.empty()) {
      Similarity<double> sim = s.top();
      sco[sim.class_name] += sim.similarity * (oob_err == 0.0 ? 1.0 : oob_err == 1.0 ? 0.0 : log((1.0-oob_err)/oob_err));
      s.pop();
    }
  }
  std::unordered_map<std::string, double>::const_iterator s_it = sco.begin();
  while (s_it != sco.end()) {
    similarities.add(s_it->first, s_it->second);
    ++s_it;
  }
  return similarities;
}

void CRF_BOOST::parse_test_line(const std::string& line){
  std::vector<std::string> tokens; tokens.reserve(100);
  Utils::string_tokenize(line, tokens, ";");
  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return;

  std::unordered_map<Inst_sp, double> doc_similarities;

  Instance<double, std::string> doc;
  doc.id(tokens[0]);
  doc.output(tokens[1]);

  for (size_t i = 2; i < tokens.size()-1; i+=2) {
    unsigned int fid = std::stoi(tokens[i]);
    double val = std::stod(tokens[i+1]);
    doc.add(fid, val);
  }

  Scores<double> similarities = classify(doc, doc_similarities);
  docs_processed_++;
//  std::cerr.precision(4);
//  std::cerr.setf(std::ios::fixed);
//  std::cerr << "\r" << docs_processed_ << ".";
  
  get_outputer()->output(similarities);
}

#endif
