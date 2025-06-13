#include "koreandictionary.h"

#include <istream>

namespace fasttext {

KoreanDictionary::KoreanDictionary(std::shared_ptr<KoreanArgs> args)
    : Dictionary(args),
      args_(args) {}

KoreanDictionary::KoreanDictionary(std::shared_ptr<KoreanArgs> args, std::istream& in)
    : Dictionary(args),
      args_(args) {
  load(in);
}

void KoreanDictionary::computeSubwords(const std::string& word, std::vector<int32_t>& ngrams, std::vector<std::string>* substrings) const {
  for (size_t i = 0, n_jamos = 0; i < word.size(); i++) {
    if ((word[i] & 0xC0) == 0x80)
      continue;

    std::string jamogram;
    for (size_t j = i, n = 1; j < word.size() && n <= args_->maxjn; n++) {
      jamogram.push_back(word[j++]);
      while (j < word.size() && (word[j] & 0xC0) == 0x80)
        jamogram.push_back(word[j++]);

      if (n >= args_->minjn && !(n == 1 && (i == 0 || j == word.size()))) {
        int32_t h = hash(jamogram) % args_->bucket;
        pushHash(ngrams, h);
        if (substrings)
          substrings->push_back(jamogram);
      }
    }
  }
  
  for (
    size_t i = word.starts_with(BOW);
    i < word.size() - word.ends_with(EOW);
    i += (word[i] & 0xF0) & 0xE0
         && ((word[i] & 0x0F) << 12 | (word[i + 1] & 0x3F) << 6 | (word[i + 2] & 0x3F)) > 0x3130
         && ((word[i] & 0x0F) << 12 | (word[i + 1] & 0x3F) << 6 | (word[i + 2] & 0x3F)) < 0x3164
         ? word[i + 3] == args_->emptyjschar
           ? 5
           : word[i + 6] == args_->emptyjschar
             ? 7
             : 9
         : word[i] == args_->emptyjschar
           ? 5
           : 1
  ) {
    std::string chargram;
    for (size_t j = i, n = 1; j < word.size() - word.ends_with(EOW) && n <= args_->maxn; n++) {
      if ((word[j] & 0xF0) & 0xE0
          && ((word[j] & 0x0F) << 12 | (word[j + 1] & 0x3F) << 6 | (word[j + 2] & 0x3F)) > 0x3130
          && ((word[j] & 0x0F) << 12 | (word[j + 1] & 0x3F) << 6 | (word[j + 2] & 0x3F)) < 0x3164
          || word[j] == args_->emptyjschar)
        for (size_t k = 0; k < 3; k++)
          if (word[j] == args_->emptyjschar)
            chargram += word[j++];
          else {
            chargram += word.substr(j,3);
            j += 3;
          }
      else
        do
          chargram += word[j++];
        while (j < word.size() && (word[j] & 0xC0) == 0x80);

      if (n >= args_->minn) {
        pushHash(ngrams, hash(chargram) % args_->bucket);
        if (substrings)
          substrings->push_back(chargram);
      }
    }
  }
}

} // namespace fasttext