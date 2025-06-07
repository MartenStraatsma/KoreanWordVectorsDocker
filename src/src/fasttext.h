/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <time.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <queue>
#include <set>
#include <tuple>

#include "args.h"
#include <fasttext/densematrix.h>
#include "dictionary.h"
#include <fasttext/matrix.h>
#include "meter.h"
#include <fasttext/model.h>
#include <fasttext/real.h>
#include <fasttext/utils.h>
#include <fasttext/vector.h>

namespace koreanfasttext {

class FastText {
 public:
  using TrainCallback =
      std::function<void(float, float, double, double, int64_t)>;

 protected:
  std::shared_ptr<Args> args_;
  std::shared_ptr<Dictionary> dict_;
  std::shared_ptr<fasttext::Matrix> input_;
  std::shared_ptr<fasttext::Matrix> output_;
  std::shared_ptr<fasttext::Model> model_;
  std::atomic<int64_t> tokenCount_{};
  std::atomic<fasttext::real> loss_{};
  std::chrono::steady_clock::time_point start_;
  bool quant_;
  int32_t version;
  std::unique_ptr<fasttext::DenseMatrix> wordVectors_;
  std::exception_ptr trainException_;

  void signModel(std::ostream&);
  bool checkModel(std::istream&);
  void startThreads(const TrainCallback& callback = {});
  void addInputVector(fasttext::Vector&, int32_t) const;
  void trainThread(int32_t, const TrainCallback& callback);
  std::vector<std::pair<fasttext::real, std::string>> getNN(
      const fasttext::DenseMatrix& wordVectors,
      const fasttext::Vector& queryVec,
      int32_t k,
      const std::set<std::string>& banSet);
  void lazyComputeWordVectors();
  void printInfo(fasttext::real, fasttext::real, std::ostream&);
  std::shared_ptr<fasttext::Matrix> getInputMatrixFromFile(const std::string&) const;
  std::shared_ptr<fasttext::Matrix> createRandomMatrix() const;
  std::shared_ptr<fasttext::Matrix> createTrainOutputMatrix() const;
  std::vector<int64_t> getTargetCounts() const;
  std::shared_ptr<fasttext::Loss> createLoss(std::shared_ptr<fasttext::Matrix>& output);
  void supervised(
      fasttext::Model::State& state,
      fasttext::real lr,
      const std::vector<int32_t>& line,
      const std::vector<int32_t>& labels);
  void cbow(fasttext::Model::State& state, fasttext::real lr, const std::vector<int32_t>& line);
  void skipgram(fasttext::Model::State& state, fasttext::real lr, const std::vector<int32_t>& line);
  std::vector<int32_t> selectEmbeddings(int32_t cutoff) const;
  void precomputeWordVectors(fasttext::DenseMatrix& wordVectors);
  bool keepTraining(const int64_t ntokens) const;
  void buildModel();
  std::tuple<int64_t, double, double> progressInfo(fasttext::real progress);

 public:
  FastText();

  int32_t getWordId(const std::string& word) const;

  int32_t getSubwordId(const std::string& subword) const;

  int32_t getLabelId(const std::string& label) const;

  void getWordVector(fasttext::Vector& vec, const std::string& word) const;

  void getSubwordVector(fasttext::Vector& vec, const std::string& subword) const;

  inline void getInputVector(fasttext::Vector& vec, int32_t ind) {
    vec.zero();
    addInputVector(vec, ind);
  }

  const Args getArgs() const;

  std::shared_ptr<const Dictionary> getDictionary() const;

  std::shared_ptr<const fasttext::DenseMatrix> getInputMatrix() const;

  void setMatrices(
      const std::shared_ptr<fasttext::DenseMatrix>& inputMatrix,
      const std::shared_ptr<fasttext::DenseMatrix>& outputMatrix);

  std::shared_ptr<const fasttext::DenseMatrix> getOutputMatrix() const;

  void saveVectors(const std::string& filename);

  void saveModel(const std::string& filename);

  void saveOutput(const std::string& filename);

  void loadModel(std::istream& in);

  void loadModel(const std::string& filename);

  void getSentenceVector(std::istream& in, fasttext::Vector& vec);

  void quantize(const Args& qargs, const TrainCallback& callback = {});

  std::tuple<int64_t, double, double>
  test(std::istream& in, int32_t k, fasttext::real threshold = 0.0);

  void test(std::istream& in, int32_t k, fasttext::real threshold, Meter& meter) const;

  void predict(
      int32_t k,
      const std::vector<int32_t>& words,
      fasttext::Predictions& predictions,
      fasttext::real threshold = 0.0) const;

  bool predictLine(
      std::istream& in,
      std::vector<std::pair<fasttext::real, std::string>>& predictions,
      int32_t k,
      fasttext::real threshold) const;

  std::vector<std::pair<std::string, fasttext::Vector>> getNgramVectors(
      const std::string& word) const;

  std::vector<std::pair<fasttext::real, std::string>> getNN(
      const std::string& word,
      int32_t k);

  std::vector<std::pair<fasttext::real, std::string>> getAnalogies(
      int32_t k,
      const std::string& wordA,
      const std::string& wordB,
      const std::string& wordC);

  void train(const Args& args, const TrainCallback& callback = {});

  void abort();

  int getDimension() const;

  bool isQuant() const;

  class AbortError : public std::runtime_error {
   public:
    AbortError() : std::runtime_error("Aborted.") {}
  };
};
} // namespace koreanfasttext
