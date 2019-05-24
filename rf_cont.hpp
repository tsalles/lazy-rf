#ifndef CRF_HPP__
#define CRF_HPP__

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
#include "weight_set.hpp"

#define NUM_THREADS 8

class CRF : public SupervisedClassifier{
  using Inst_sp = std::shared_ptr<Instance<double, std::string>>;

  public:
    CRF(unsigned int r, double m=1.0, unsigned int num_trees=10, unsigned int maxh=0, bool trn_err=false) : SupervisedClassifier(r), num_trees_(num_trees), m_(m), doc_delete_(true), maxh_(maxh), trn_err_(trn_err) { trees_.reserve(num_trees); total_oob_ = 0.0; srand(time(NULL)); oob_.resize(num_trees); }
    ~CRF();
    bool parse_train_line(const std::string&);
    void train(const std::string&);
    void parse_test_line(const std::string&);
    void reset_model();
    void add_document(Inst_sp);
    void add_document_bag(std::set<Inst_sp> &);
    WeightSet<std::string> *build(WeightSet<std::string> * w = NULL);
    void set_doc_delete(const bool&);
    Scores<double> classify(Instance<double, std::string> &);
    double avg_oob_err() { return (oob_err_.size() > 0) ? total_oob_/oob_err_.size() : 0.0; }
  private:
    std::vector<CDT*> trees_;
    std::vector<Inst_sp> docs_;
    std::vector<std::vector<Inst_sp>> oob_;
    double total_oob_;
    std::vector<double> oob_err_;
    double m_;
    unsigned int num_trees_;
    bool doc_delete_;
    unsigned int maxh_;
    bool trn_err_;
};

CRF::~CRF(){
  reset_model();
}

void CRF::set_doc_delete(const bool& dd){
  doc_delete_ = dd;
}

void CRF::reset_model(){
  for(int i = 0; i < num_trees_; i++){
    delete trees_[i];
  }
  trees_.clear();
  if(doc_delete_){
    docs_.clear();
    oob_.clear();
    total_oob_ = 0.0;
  }
  m_ = 1.0;
  num_trees_ = 0;
}

bool CRF::parse_train_line(const std::string& line){
  std::vector<std::string> tokens;
  Utils::string_tokenize(line, tokens, ";");
  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return false;
  Inst_sp doc = Inst_sp(new Instance<double, std::string>());
  std::string doc_id = tokens[0];
  doc->id(doc_id);
  doc->output(tokens[1]);
  for (size_t i = 2; i < tokens.size()-1; i+=2) {
    unsigned int fid = std::stoi(tokens[i]);
    double val = tokens[i+1] == "?" ? 0.0 : std::stod(tokens[i+1]);
    doc->add(fid, val);
  }
  for (unsigned int i = 0; i < num_trees_; i++) {
    oob_[i].push_back(doc);
  }
  docs_.push_back(doc);
  return true;
}

void CRF::train(const std::string& train_fn){
  SupervisedClassifier::train(train_fn);
  build();
}

void CRF::add_document(Inst_sp doc){
  docs_.push_back(doc);
  for (unsigned int i = 0; i < num_trees_; i++) {
    oob_[i].push_back(doc);
  } 
}

void CRF::add_document_bag(std::set<Inst_sp>& bag){
  std::set<Inst_sp>::const_iterator cIt = bag.begin();
  while(cIt != bag.end()){
    docs_.push_back(*cIt);
    ++cIt;
  }
}

// Should return
// 1) Map: out-of-bag sample ID -> bool misclassified ?
WeightSet<std::string> * CRF::build(WeightSet<std::string> *w) {
  const unsigned int docs_size = docs_.size();
  #pragma omp parallel for schedule(static)
  for(int i = 0; i < num_trees_; i++) {
    Items<double, std::string> bag;
    for(int j = 0; j < docs_size; j++) {
      #pragma omp critical (oob_update)
      {
        unsigned int rndIdx = rand() % docs_size;
        bag.add(docs_[rndIdx]);
        if (!trn_err_) {
          oob_[i][rndIdx] = NULL; // it isn't an oob sample
        }
      }
    }
    trees_[i] = new CDT(bag, round, maxh_, m_);
    trees_[i]->build();

    // evaluate OOB error
    double miss = 0.0, total = 0.0;
    std::map<unsigned int, bool> is_miss;
    for (unsigned int oobidx = 0; oobidx < oob_[i].size(); oobidx++) {
      if (oob_[i][oobidx] != NULL) {
        std::unordered_map<std::string, unsigned int> partial_scores = trees_[i]->classify(*(oob_[i][oobidx]));
        // see if its a missclassification
        double max = -9999.99;
        std::string maxCl;
        std::unordered_map<std::string, unsigned int>::const_iterator cIt = partial_scores.begin();
        while(cIt != partial_scores.end()) {
          if (cIt->second > max) {
            maxCl = cIt->first;
            max = cIt->second;
          }
          ++cIt;
        }
        if (maxCl != oob_[i][oobidx]->output()) {
          is_miss[oobidx] = true;
          miss++;// += (w != NULL) ? w->get(oob_[i][oobidx]->get_id()) : 1.0;
        }
        total++;// += (w != NULL) ? w->get(oob_[i][oobidx]->get_id()) : 1.0;
      }
    }
    double oob_err = total == 0.0 ? 0.0 : (miss / total);
    double alpha = oob_err == 0.0 ? 1.0 : oob_err == 1.0 ? 0.0 : log((1.0-oob_err)/oob_err);

    #pragma omp critical (oob_update)
    {
    if (w != NULL) { 
      {
        for (unsigned int oobidx = 0; oobidx < oob_[i].size(); oobidx++) {
          if (oob_[i][oobidx] != NULL) {
            w->set(oob_[i][oobidx]->id(),  w->get(oob_[i][oobidx]->id()) * exp((is_miss[oobidx] ? -1.0 : 1.0) * alpha));
            //std::cerr << "i=" << i << " oobidx=" << oobidx << " w=" << w->get(oob_[i][oobidx]->get_id()) << " alpha=" << alpha << std::endl;
          }
        }
      }
    }
    oob_err_.push_back(oob_err);
    total_oob_ += oob_err;
    }

    //std::cerr << "trr_oob[" << i << "] = " << miss << "/" << total << "=" << oob_err << std::endl;
  }
}

Scores<double> CRF::classify(Instance<double, std::string> &doc){
  Scores<double> similarities(doc.id(), doc.output());
  std::map<std::string, double> scores;
  std::map<std::string, unsigned int> trees_count;
  for(int i = 0; i < num_trees_; i++) {
    std::unordered_map<std::string, unsigned int> partial_scores = trees_[i]->classify(doc);
    std::unordered_map<std::string, unsigned int>::const_iterator cIt = partial_scores.begin();
    while(cIt != partial_scores.end()) {
      std::map<std::string, double>::iterator it = scores.find(cIt->first);
      if(it == scores.end()) {
        it = (scores.insert(std::make_pair(cIt->first, 0.0))).first;
      }
      double weight = oob_err_[i];
      it->second += cIt->second * (weight == 0.0 ? 1.0 : weight == 1.0 ? 0.0 : log((1.0-weight)/weight));

      std::map<std::string, unsigned int>::iterator it_tc = trees_count.find(cIt->first);
      if(it_tc == trees_count.end()){
        it_tc = (trees_count.insert(std::make_pair(cIt->first, 0))).first;
      }
      (it_tc->second)++;

      ++cIt;
    }
  }
  std::map<std::string, double>::const_iterator cIt_s = scores.begin();
  while(cIt_s != scores.end()){
    std::map<std::string, unsigned int>::const_iterator cIt_t = trees_count.find(cIt_s->first);
    if(cIt_t == trees_count.end()) {
      similarities.add(cIt_s->first, 0.0);
    }
    //else if(cIt_t->second != 1){
    //  similarities.add(cIt_s->first, cIt_s->second / log(cIt_t->second));
    //}
    else{
      similarities.add(cIt_s->first, cIt_s->second);// / (1.0 + log(cIt_t->second)));
    }
    ++cIt_s;
  }
  return similarities;
}

void CRF::parse_test_line(const std::string& line){
  std::vector<std::string> tokens;
  Utils::string_tokenize(line, tokens, ";");

  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return;

  Instance<double, std::string> doc;
  std::string doc_id = tokens[0];
  doc.id(doc_id);
  std::string doc_class = tokens[1];
  doc.output(doc_class);
  for (size_t i = 2; i < tokens.size()-1; i+=2) {
    unsigned int fid = std::stoi(tokens[i]);
    double val = std::stod(tokens[i+1]);
    doc.add(fid, val);
  }

  Scores<double> similarities = classify(doc);
  
  get_outputer()->output(similarities);
}

#endif
