#pragma once

#include "args.h"

namespace fasttext {

class KoreanArgs : public Args {
 public:
  KoreanArgs();
  int minjn;
  int maxjn;
  char emptyjschar;

  void parseArgs(const std::vector<std::string>& args);
  void printDictionaryHelp() override;
  void save(std::ostream&) override;
  void load(std::istream&) override;
  void dump(std::ostream&) const override;
};

} // namespace fasttext