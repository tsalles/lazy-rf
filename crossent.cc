#include <iostream>
#include <fstream>
#include <cmath>
#include <limits>

#include <unordered_map>
#include <unordered_set>

#include <vector>
#include <set>

// Maximum Fisher’s discriminant ratio:
// This measure computes the maximum discriminative
// power of each attribute.
// F1 = max_a f_a,
// where 1 \leq a \leq l denotes the input attributes,
// f_a is the discriminant ratio of attribute a.
// For m classes, f_a is given by:
// \frac {\sum_{c_i,c_j} p(c_i) p(c_j) (\mu_{c_i}^a - \mu_{c_j}^a)^2}
//       {\sum_{c_i} p(c_i) \sigma_{c_i}^2},
// \forall c_i \neq c_j.

class FisherDR {

 public:
  double compute(const std::string &fn) {
    clear();
    if (parse(fn)) return f1Measure();
    std::cerr << "I was unable to parse input file '"
              << fn << "'." << std::endl;
    return -1;
  }

 private:
  std::set<std::string> cl_;
  std::unordered_map<std::string, double> nc_;
  std::unordered_map<std::string, std::unordered_map<unsigned int, double>> sum_attr_;
  std::unordered_map<std::string, std::unordered_map<unsigned int, double>> s_sum_attr_;
  std::unordered_map<std::string, std::unordered_map<unsigned int, double>> n_attr_;
  std::unordered_set<unsigned int> attr_;
  unsigned int n_;

  void clear() {
    cl_.clear();
    sum_attr_.clear();
    s_sum_attr_.clear();
    n_attr_.clear();
    attr_.clear();
    n_ = 0;
  }

  double variance(const std::string &c, const unsigned int f) {
    double n = n_attr_[c][f];
    double s = sum_attr_[c][f];
    double ss = s_sum_attr_[c][f];
    return n > 0.0 ? (ss - (pow(s, 2.0)/n))/n : 0.0;
  }

  double mean(const std::string &c, const unsigned int f) {
    double n = n_attr_[c][f];
    double s = sum_attr_[c][f];
    return n > 0 ? s / n : 0.0;
  }

  double f1Measure() {
    double sum = 0.0, n = 0.0;
    for (unsigned int a : attr_) {
      sum += f1Measure(a);
      n++;
    }
    return n > 0 ? sum/n : 0.0;
  }

  double f1Measure(unsigned int a) {
//    std::cerr << "[" << a << "]" << std::endl;
    double num = 0.0, den = 0.0;
    std::set<std::string>::const_iterator ci_it = cl_.begin();
    std::set<std::string>::const_iterator cj_it = ci_it;
    while (ci_it != cl_.end() && cj_it != cl_.end()) {
      double pi = nc_[*ci_it]/n_;
      double mi = mean(*ci_it, a);
      double si = variance(*ci_it, a);

//      std::cerr << "--[" << *ci_it << "]--" << std::endl; 
//      std::cerr << "   pi=" << pi << std::endl;
//      std::cerr << "   mi=" << mi << std::endl;
//      std::cerr << "   si=" << si << std::endl;
      cj_it = ci_it; ++cj_it;
      while (cj_it != cl_.end()) {     
        double pj = nc_[*cj_it]/n_;
        double mj = mean(*cj_it, a);
        double sj = variance(*cj_it, a);
        double p_num = (pi*pj*pow((mi-mj), 2.0));
        double p_den = (pi*pj*(1.0+(si*sj)));
//        std::cerr << "  --[" << *cj_it << "]--" << std::endl;
//        std::cerr << "     pj=" << pj << std::endl;
//        std::cerr << "     mj=" << pj << std::endl;
//        std::cerr << "     sj=" << sj << std::endl;
//        std::cerr << "    n +=" << p_num << std::endl;
//        std::cerr << "    d +=" << p_den << std::endl;
        num += p_num;
        den += p_den;
        ++cj_it;
      }
      //den += (pi*si);
//      std::cerr << "  d +=" << (pi*si) << std::endl;
      ++ci_it;
    }

    return num == 0.0 ? 0.0 : den > 0.0 ? num/den : std::numeric_limits<double>::infinity();
  }

  bool parse(const std::string &fn);

  void tokenize(const std::string &str,
                std::vector<std::string> &tokens,
                const std::string &delimiters = " ") {
    std::string::size_type lastPos =
        str.find_first_not_of(delimiters, 0);
    std::string::size_type pos =
        str.find_first_of(delimiters, lastPos);
    while (std::string::npos != pos || std::string::npos != lastPos) {
      tokens.push_back(str.substr(lastPos, pos - lastPos));
      lastPos = str.find_first_not_of(delimiters, pos);
      pos = str.find_first_of(delimiters, lastPos);
    }
  }

};


bool FisherDR::parse(const std::string &fn) {

  std::ifstream f(fn.data());
  if (f) {
    std::string ln;
    std::set<std::string> attributes;
    while (std::getline(f, ln)) {
      std::vector<std::string> tks;
      tokenize(ln, tks, ";");
      std::string c = tks[1];
      cl_.insert(c);
      nc_[c]++;
      for (unsigned int i = 2; i < tks.size()-1; i+=2) {
        unsigned int a = std::stoi(tks[i]);
        double w = std::stod(tks[i+1]);
        sum_attr_[c][a] += w;
        s_sum_attr_[c][a] += w*w;
        n_attr_[c][a]++;
        attr_.insert(a);
      }
      n_++;
    }

    f.close();
    std::cerr << "Summary" << std::endl;
    std::cerr << "sum_attr_.size()   = " <<   sum_attr_.size() << std::endl;
    std::cerr << "s_sum_attr_.size() = " << s_sum_attr_.size() << std::endl;
    std::cerr << "n_attr_.size()     = " <<     n_attr_.size() << std::endl;
    std::cerr << "attr_.size()       = " <<       attr_.size() << std::endl;
    std::cerr << "                n_ = " <<                 n_ << std::endl;


  } else {
    return false;
  }

  return true;
}

int main(int argc, char **argv) {

  if (argc < 2) {
    std::cerr << "[USAGE] " << argv[0] << " <input>" << std::endl;
    return 1; 
  }

  FisherDR fdr;
  double res = fdr.compute(std::string(argv[1]));
  std::cout << res << std::endl;
 
  return 0;
}
