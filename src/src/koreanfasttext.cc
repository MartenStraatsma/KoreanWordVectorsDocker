#include "koreanfasttext.h"
#include "loss.h"
#include "quantmatrix.h"

#include <istream>
#include <stdexcept>

namespace fasttext {

KoreanFastText::KoreanFastText() : FastText() {}

std::shared_ptr<const KoreanDictionary> KoreanFastText::getDictionary() const {
  return std::static_pointer_cast<KoreanDictionary>(dict_);
}

const KoreanArgs KoreanFastText::getArgs() const {
  return *std::static_pointer_cast<KoreanArgs>(args_).get();
}

void KoreanFastText::loadModel(std::istream& in) {
  args_ = std::make_shared<KoreanArgs>();
  input_ = std::make_shared<DenseMatrix>();
  output_ = std::make_shared<DenseMatrix>();
  args_->load(in);
  if (version == 11 && args_->model == model_name::sup)
    // backward compatibility: old supervised models do not use char ngrams.
    args_->maxn = 0;
  dict_ = std::make_shared<KoreanDictionary>(std::static_pointer_cast<KoreanArgs>(args_), in);

  bool quant_input;
  in.read((char*)&quant_input, sizeof(bool));
  if (quant_input) {
    quant_ = true;
    input_ = std::make_shared<QuantMatrix>();
  }
  input_->load(in);

  if (!quant_input && dict_->isPruned())
    throw std::invalid_argument(
        "Invalid model file.\n"
        "Please download the updated model from www.fasttext.cc.\n"
        "See issue #332 on Github for more information.\n");

  in.read((char*)&args_->qout, sizeof(bool));
  if (quant_ && args_->qout)
    output_ = std::make_shared<QuantMatrix>();
  output_->load(in);

  buildModel();
}

void KoreanFastText::train(const KoreanArgs& args, const TrainCallback& callback) {
  args_ = std::make_shared<KoreanArgs>(args);
  dict_ = std::make_shared<KoreanDictionary>(std::static_pointer_cast<KoreanArgs>(args_));
  if (args_->input == "-")
    // manage expectations
    throw std::invalid_argument("Cannot use stdin for training!");

  std::ifstream ifs(args_->input);
  if (!ifs.is_open())
    throw std::invalid_argument(args_->input + " cannot be opened for training!");

  dict_->readFromFile(ifs);
  ifs.close();
  if (!args_->pretrainedVectors.empty()) {
    input_ = getInputMatrixFromFile(args_->pretrainedVectors);
  } else
    input_ = createRandomMatrix();
  output_ = createTrainOutputMatrix();
  quant_ = false;
  auto loss = createLoss(output_);
  bool normalizeGradient = (args_->model == model_name::sup);
  model_ = std::make_shared<Model>(input_, output_, loss, normalizeGradient);
  startThreads(callback);
}

} // namespace fasttext