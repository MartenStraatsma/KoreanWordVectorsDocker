#include "koreanargs.h"
#include "dictionary.h"

namespace fasttext {

class KoreanDictionary : public Dictionary {
 protected:
  std::shared_ptr<KoreanArgs> args_;

 public:
  explicit KoreanDictionary(std::shared_ptr<KoreanArgs>);
  explicit KoreanDictionary(std::shared_ptr<KoreanArgs>, std::istream&);
  void computeSubwords(const std::string&, std::vector<int32_t>&, std::vector<std::string>* substrings = nullptr) const override;
};

} // namespace fasttext