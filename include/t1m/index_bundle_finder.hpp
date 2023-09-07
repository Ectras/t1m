#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>

namespace t1m::internal
{
  class IndexBundleFinder
  {
  public:
    IndexBundleFinder(std::vector<int> labelsA, std::vector<int> labelsB, std::vector<int> labelsC)
        : labelsA(std::move(labelsA)), labelsB(std::move(labelsB)), labelsC(std::move(labelsC))
    {
      this->find();
      this->find_c_permutation();
    }

  private:
    void find()
    {
      bool in_I;
      std::unordered_set<int> setB{labelsB.cbegin(), labelsB.cend()};

      for (size_t i = 0; i < labelsA.size(); i++)
      {
        in_I = false;
        for (size_t j = 0; j < labelsB.size(); j++)
        {
          if (labelsA.at(i) == labelsB.at(j))
          {
            in_I = true;
            this->Pa.push_back(i);
            this->Pb.push_back(j);
            setB.erase(labelsA.at(i));
          }
        }

        if (!in_I)
          this->I.push_back(i);
      }

      for (int j = 0; j < labelsB.size(); j++)
        if (setB.count(labelsB.at(j)) > 0)
        {
          this->J.push_back(j);
        }
    }

    void find_c_permutation()
    {
      for (const auto &idx : this->I)
      {
        for (int j = 0; j < this->labelsC.size(); j++)
        {
          if (this->labelsA.at(idx) == this->labelsC.at(j))
          {
            this->Ic.push_back(j);
          }
        }
      }

      for (const auto &idx : this->J)
      {
        for (int j = 0; j < this->labelsC.size(); j++)
        {
          if (this->labelsB.at(idx) == this->labelsC.at(j))
          {
            this->Jc.push_back(j);
          }
        }
      }
    }

  public:
    std::vector<size_t> I;
    std::vector<size_t> J;
    std::vector<size_t> Pa;
    std::vector<size_t> Pb;

    std::vector<size_t> Ic;
    std::vector<size_t> Jc;

  private:
    std::vector<int> labelsA;
    std::vector<int> labelsB;
    std::vector<int> labelsC;
  };
};
