#include "koreanargs.h"

#include <cstdlib>

#include <iostream>
#include <stdexcept>
#include <string>

namespace fasttext {

KoreanArgs::KoreanArgs() : Args(),
  minjn(3),
  maxjn(5),
  emptyjschar('e')
{
  bucket = 10000000;
  minn = 2;
  maxn = 4;
}

void KoreanArgs::parseArgs(const std::vector<std::string>& args) {
  const std::string& command(args[1]);
  if (command == "supervised") {
    model = model_name::sup;
    loss = loss_name::softmax;
    minCount = 1;
    minn = 0;
    minjn = 0;
    maxn = 0;
    maxjn = 0;
    lr = 0.1;
  } else if (command == "cbow")
    model = model_name::cbow;

  for (int ai = 2; ai < args.size(); ai += 2) {
    if (args[ai][0] != '-') {
      std::cerr << "Provided argument without a dash! Usage:" << std::endl;
      printHelp();
      exit(EXIT_FAILURE);
    }

    try {
      setManual(args[ai].substr(1));

      if (args[ai] == "-h") {
        std::cerr << "Here is the help! Usage:" << std::endl;
        printHelp();
        exit(EXIT_FAILURE);
      } else if (args[ai] == "-input")
        input = std::string(args.at(ai + 1));
      else if (args[ai] == "-output")
        output = std::string(args.at(ai + 1));
      else if (args[ai] == "-emptyjschar") {
        if (args.at(ai + 1).size() == 1)
          emptyjschar = args.at(ai + 1).front();
        else {
          std::cerr << "Empty jongsung character must be a single character." << std::endl;
          printHelp();
          exit(EXIT_FAILURE);
        }
      }
      else if (args[ai] == "-lr")
        lr = std::stof(args.at(ai + 1));
      else if (args[ai] == "-lrUpdateRate")
        lrUpdateRate = std::stoi(args.at(ai + 1));
      else if (args[ai] == "-dim")
        dim = std::stoi(args.at(ai + 1));
      else if (args[ai] == "-ws")
        ws = std::stoi(args.at(ai + 1));
      else if (args[ai] == "-epoch")
        epoch = std::stoi(args.at(ai + 1));
      else if (args[ai] == "-minCount")
        minCount = std::stoi(args.at(ai + 1));
      else if (args[ai] == "-minCountLabel")
        minCountLabel = std::stoi(args.at(ai + 1));
      else if (args[ai] == "-neg")
        neg = std::stoi(args.at(ai + 1));
      else if (args[ai] == "-wordNgrams")
        wordNgrams = std::stoi(args.at(ai + 1));
      else if (args[ai] == "-loss") {
        if (args.at(ai + 1) == "hs")
          loss = loss_name::hs;
        else if (args.at(ai + 1) == "ns")
          loss = loss_name::ns;
        else if (args.at(ai + 1) == "softmax")
          loss = loss_name::softmax;
        else if (args.at(ai + 1) == "one-vs-all" || args.at(ai + 1) == "ova")
          loss = loss_name::ova;
        else {
          std::cerr << "Unknown loss: " << args.at(ai + 1) << std::endl;
          printHelp();
          exit(EXIT_FAILURE);
        }
      } else if (args[ai] == "-bucket")
        bucket = std::stoi(args.at(ai + 1));
      else if (args[ai] == "-minn")
        minn = std::stoi(args.at(ai + 1));
      else if (args[ai] == "-maxn")
        maxn = std::stoi(args.at(ai + 1));
      else if (args[ai] == "-minjn")
        minjn = std::stoi(args.at(ai + 1));
      else if (args[ai] == "-maxjn")
        maxjn = std::stoi(args.at(ai + 1));
      else if (args[ai] == "-thread")
        thread = std::stoi(args.at(ai + 1));
      else if (args[ai] == "-t")
        t = std::stof(args.at(ai + 1));
      else if (args[ai] == "-label")
        label = std::string(args.at(ai + 1));
      else if (args[ai] == "-verbose")
        verbose = std::stoi(args.at(ai + 1));
      else if (args[ai] == "-pretrainedVectors")
        pretrainedVectors = std::string(args.at(ai + 1));
      else if (args[ai] == "-saveOutput") {
        saveOutput = true; ai--;
      } else if (args[ai] == "-seed")
        seed = std::stoi(args.at(ai + 1));
      else if (args[ai] == "-qnorm") {
        qnorm = true; ai--;
      } else if (args[ai] == "-retrain") {
        retrain = true; ai--;
      } else if (args[ai] == "-qout") {
        qout = true; ai--;
      } else if (args[ai] == "-cutoff")
        cutoff = std::stoi(args.at(ai + 1));
      else if (args[ai] == "-dsub")
        dsub = std::stoi(args.at(ai + 1));
      else if (args[ai] == "-autotune-validation")
        autotuneValidationFile = std::string(args.at(ai + 1));
      else if (args[ai] == "-autotune-metric") {
        autotuneMetric = std::string(args.at(ai + 1));
        getAutotuneMetric(); // throws exception if not able to parse
        getAutotuneMetricLabel(); // throws exception if not able to parse
      } else if (args[ai] == "-autotune-predictions")
        autotunePredictions = std::stoi(args.at(ai + 1));
      else if (args[ai] == "-autotune-duration")
        autotuneDuration = std::stoi(args.at(ai + 1));
      else if (args[ai] == "-autotune-modelsize")
        autotuneModelSize = std::string(args.at(ai + 1));
      else {
        std::cerr << "Unknown argument: " << args[ai] << std::endl;
        printHelp();
        exit(EXIT_FAILURE);
      }

    } catch (std::out_of_range) {
      std::cerr << args[ai] << " is missing an argument" << std::endl;
      printHelp();
      exit(EXIT_FAILURE);
    }
  }
  if (input.empty() || output.empty()) {
    std::cerr << "Empty input or output path." << std::endl;
    printHelp();
    exit(EXIT_FAILURE);
  }

  if (wordNgrams <= 1 && maxn == 0 && !hasAutotune())
    bucket = 0;
}

void KoreanArgs::printDictionaryHelp() {
  std::cerr << "\nThe following arguments for the dictionary are optional:\n"
            << "  -minCount           minimal number of word occurences [" << minCount << "]\n"
            << "  -minCountLabel      minimal number of label occurences [" << minCountLabel << "]\n"
            << "  -wordNgrams         max length of word ngram [" << wordNgrams << "]\n"
            << "  -bucket             number of buckets [" << bucket << "]\n"
            << "  -minn               min length of char ngram [" << minn << "]\n"
            << "  -maxn               max length of char ngram [" << maxn << "]\n"
            << "  -minjn              min length of jamo ngram [" << minjn << "]\n"
            << "  -maxjn              max length of jamo ngram [" << maxjn << "]\n"
            << "  -emptyjschar        empty jongsung symbol [" << emptyjschar << "]\n" 
            << "  -t                  sampling threshold [" << t << "]\n"
            << "  -label              labels prefix [" << label << "]\n";
}

void KoreanArgs::save(std::ostream& out) {
  out.write((char*)&(dim), sizeof(int));
  out.write((char*)&(ws), sizeof(int));
  out.write((char*)&(epoch), sizeof(int));
  out.write((char*)&(minCount), sizeof(int));
  out.write((char*)&(neg), sizeof(int));
  out.write((char*)&(wordNgrams), sizeof(int));
  out.write((char*)&(loss), sizeof(loss_name));
  out.write((char*)&(model), sizeof(model_name));
  out.write((char*)&(bucket), sizeof(int));
  out.write((char*)&(minn), sizeof(int));
  out.write((char*)&(maxn), sizeof(int));
  out.write((char*)&(emptyjschar), sizeof(char));
  out.write((char*)&(minjn), sizeof(int));
  out.write((char*)&(maxjn), sizeof(int));
  out.write((char*)&(lrUpdateRate), sizeof(int));
  out.write((char*)&(t), sizeof(double));
}

void KoreanArgs::load(std::istream& in) {
  in.read((char*)&(dim), sizeof(int));
  in.read((char*)&(ws), sizeof(int));
  in.read((char*)&(epoch), sizeof(int));
  in.read((char*)&(minCount), sizeof(int));
  in.read((char*)&(neg), sizeof(int));
  in.read((char*)&(wordNgrams), sizeof(int));
  in.read((char*)&(loss), sizeof(loss_name));
  in.read((char*)&(model), sizeof(model_name));
  in.read((char*)&(bucket), sizeof(int));
  in.read((char*)&(minn), sizeof(int));
  in.read((char*)&(maxn), sizeof(int));
  in.read((char*)&(emptyjschar), sizeof(char));
  in.read((char*)&(minjn), sizeof(int));
  in.read((char*)&(maxjn), sizeof(int));
  in.read((char*)&(lrUpdateRate), sizeof(int));
  in.read((char*)&(t), sizeof(double));
}

void KoreanArgs::dump(std::ostream& out) const {
  out << "dim " << dim << std::endl;
  out << "ws " << ws << std::endl;
  out << "epoch " << epoch << std::endl;
  out << "minCount " << minCount << std::endl;
  out << "neg " << neg << std::endl;
  out << "wordNgrams " << wordNgrams << std::endl;
  out << "loss " << lossToString(loss) << std::endl;
  out << "model " << modelToString(model) << std::endl;
  out << "bucket " << bucket << std::endl;
  out << "minn " << minn << std::endl;
  out << "maxn " << maxn << std::endl;
  out << "emptyjschar " << emptyjschar << std::endl;
  out << "minjn " << minjn << std::endl;
  out << "maxjn " << maxjn << std::endl;
  out << "lrUpdateRate " << lrUpdateRate << std::endl;
  out << "t " << t << std::endl;
}

} // namespace fasttext