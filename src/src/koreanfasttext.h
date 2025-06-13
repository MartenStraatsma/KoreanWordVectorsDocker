#include "koreanargs.h"
#include "koreandictionary.h"
#include "fasttext.h"

namespace fasttext {

class KoreanFastText : public FastText {
 public:
  using FastText::loadModel;

 public:
  KoreanFastText();

  const KoreanArgs getArgs() const;

  std::shared_ptr<const KoreanDictionary> getDictionary() const;

  void loadModel(std::istream& in) override;

  void train(const KoreanArgs& args, const TrainCallback& callback = {});
};

} // namespace fasttext