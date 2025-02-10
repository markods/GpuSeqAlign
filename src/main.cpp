#include "json.hpp"
#include <chrono>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

#include "common.hpp"
#include "nw-algorithm.hpp"
using json = nlohmann::ordered_json;
using namespace std::string_literals;

// main program
int main(int argc, char *argv[])
{
    // check if the arguments are valid
    if (argc != 4)
    {
        std::cerr << "ERR - invalid arguments\n"
                     "nw fsubsts fparams fseqs\n"
                     "   fsubst     json file with substitution matrices\n"
                     "   fparams    json file with nw parameters\n"
                     "   fseqs      json file with sequences to be compared\n";

        exit(-1);
    }

    extern NwAlgorithmData algData;
    NwSubstData substData;
    NwParamData paramData;
    NwSeqData seqData;
    NwResData resData;
    std::ofstream ofsRes;

    resData.projPath = std::filesystem::current_path().string() + "/../../";
    resData.resrcPath = resData.projPath + "resrc/";
    resData.resPath = resData.projPath + "log/";

    resData.isoTime = IsoTime();
    resData.substFname = argv[1];
    resData.paramFname = argv[2];
    resData.seqFname = argv[3];
    resData.resFname = resData.isoTime + ".csv";

    // read data from input .json files
    // +   also open the output file
    {
        std::string substPath = resData.resrcPath + resData.substFname;
        std::string paramPath = resData.resrcPath + resData.paramFname;
        std::string seqPath = resData.resrcPath + resData.seqFname;
        std::string resPath = resData.resPath + resData.resFname;

        if (NwStat::success != readFromJson(substPath, substData))
        {
            std::cerr << "ERR - could not open/read json from substs file";
            exit(-1);
        }
        if (NwStat::success != readFromJson(paramPath, paramData))
        {
            std::cerr << "ERR - could not open/read json from params file";
            exit(-1);
        }
        if (NwStat::success != readFromJson(seqPath, seqData))
        {
            std::cerr << "ERR - could not open/read json from seqs file";
            exit(-1);
        }
        if (NwStat::success != openOutFile(resPath, ofsRes))
        {
            std::cerr << "ERR - could not open csv results file";
            exit(-1);
        }
    }

    // get the device properties
    cudaDeviceProp deviceProps;
    if (cudaSuccess != (cudaStatus = cudaGetDeviceProperties(&deviceProps, 0 /*deviceId*/)))
    {
        std::cerr << "ERR - could not get device properties";
        exit(-1);
    }

    // number of streaming multiprocessors (sm-s) and threads in a warp
    const int sm_count = deviceProps.multiProcessorCount;
    const int warpsz = deviceProps.warpSize;
    const int maxThreadsPerBlock = deviceProps.maxThreadsPerBlock;

    NwInput nw{
        ////// host specific memory
        // subst;   <-- once
        // seqX;    <-- loop-inited
        // seqY;    <-- loop-inited
        // score;   <-- algorithm-reserved

        ////// device specific memory
        // subst_gpu;   <-- once
        // seqX_gpu;    <-- algorithm-reserved
        // seqY_gpu;    <-- algorithm-reserved
        // score_gpu;   <-- algorithm-reserved
        // // sparse representation of the score matrix
        // tileHrowMat_gpu;   <-- algorithm-reserved
        // tileHcolMat_gpu;   <-- algorithm-reserved

        ////// alignment parameters
        // substsz;   <-- once
        // adjrows;   <-- loop-inited
        // adjcols;   <-- loop-inited
        // indel;   <-- once
        // // sparse representation of the score matrix
        // tileHdrMatRows;   <-- algorithm-reserved
        // tileHdrMatCols;   <-- algorithm-reserved
        // tileHrowLen;   <-- algorithm-reserved
        // tileHcolLen;   <-- algorithm-reserved

        ////// device parameters
        // sm_count;
        // warpsz;
        // maxThreadsPerBlock;
    };

    // initialize the device parameters
    nw.sm_count = sm_count;
    nw.warpsz = warpsz;
    nw.maxThreadsPerBlock = maxThreadsPerBlock;

    // initialize the substitution matrix on the cpu and gpu
    {
        nw.subst = substData.substMap[seqData.substName];
        nw.substsz = (int)std::sqrt(nw.subst.size());

        // reserve space in the gpu global memory
        try
        {
            nw.subst_gpu.init(nw.substsz * nw.substsz);
        }
        catch (const std::exception &)
        {
            std::cerr << "ERR - could not reserve space for the substitution matrix in the gpu";
            exit(-1);
        }

        // transfer the substitution matrix to the gpu global memory
        if (cudaSuccess != (cudaStatus = memTransfer(nw.subst_gpu, nw.subst, nw.substsz * nw.substsz)))
        {
            std::cerr << "ERR - could not transfer substitution matrix to the gpu";
            exit(-1);
        }
    }

    // initialize the indel cost
    nw.indel = seqData.indel;
    // initialize the letter map
    std::map<std::string, int> letterMap = substData.letterMap;

    // initialize the sequence map
    std::vector<std::vector<int>> seqList{};
    for (auto &charSeq : seqData.seqList)
    {
        auto seq = seqStrToVect(charSeq, letterMap, true /*addHeader*/);
        seqList.push_back(seq);
    }

    // initialize the gold result map (as calculated by the first algorithm)
    NwCompareData compareData{};

    // write the csv file's header
    resHeaderToCsv(ofsRes, resData);
    ofsRes.flush();

    // for all algorithms which have parameters in the param map
    for (auto &paramTuple : paramData.paramMap)
    {
        // if the current algorithm doesn't exist, skip it
        const std::string &algName = paramTuple.first;
        if (algData.algMap.find(algName) == algData.algMap.end())
        {
            continue;
        }

        std::cout << algName << ":";

        // get the current algorithm and initialize its parameters
        NwAlgorithm &alg = algData.algMap[algName];
        alg.init(paramTuple.second /*algParams*/);

        // for all Y sequences + for all X sequences (also compare every sequence with itself)
        for (int iY = 0; iY < seqList.size(); iY++)
        {
            std::cout << std::endl
                      << "|";

            for (int iX = iY; iX < seqList.size(); iX++)
            {
                // get the Y sequence
                // NOTE: the padding (zeroth element) was already added to the sequence
                nw.seqY = seqList[iY];
                nw.adjrows = (int)nw.seqY.size();

                // get the X sequence
                // NOTE: the padding (zeroth element) was already added to the sequence
                nw.seqX = seqList[iX];
                nw.adjcols = (int)nw.seqX.size();

                // if the number of columns is less than the number of rows, swap them and the sequences
                if (nw.adjcols < nw.adjrows)
                {
                    std::swap(nw.adjcols, nw.adjrows);
                    std::swap(nw.seqX, nw.seqY);
                }

                // for all parameter combinations
                for (; alg.alignPr().hasCurr(); alg.alignPr().next())
                {
                    // results from multiple repetitions
                    std::vector<NwResult> resList{};

                    // for all requested repeats
                    for (int iR = 0; iR < seqData.repeat; iR++)
                    {
                        // initialize the result in the result list
                        resList.push_back(NwResult{
                            algName,                  // algName;
                            alg.alignPr().snapshot(), // algParams;

                            nw.seqX.size(), // seqX_len;
                            nw.seqY.size(), // seqY_len;

                            iX,             // iX;
                            iY,             // iY;
                            seqData.repeat, // reps;

                            {}, // sw_align;
                            {}, // sw_hash;
                            {}, // sw_trace;

                            {}, // score_hash;
                            {}, // trace_hash;

                            {}, // errstep;   // 0 for success
                            {}, // stat;      // 0 for success
                            {}, // cudaerr;   // 0 for success
                        });
                        // get the result from the list
                        NwResult &res = resList.back();

                        // compare the sequences, hash and trace the score matrices, and verify the soundness of the results
                        if (!res.errstep && NwStat::success != (res.stat = alg.align(nw, res)))
                        {
                            res.errstep = 1;
                            res.cudaerr = cudaStatus;
                        }
                        if (!res.errstep && NwStat::success != (res.stat = alg.hash(nw, res)))
                        {
                            res.errstep = 2;
                            res.cudaerr = cudaStatus;
                        }
                        if (!res.errstep && NwStat::success != (res.stat = alg.trace(nw, res)))
                        {
                            res.errstep = 3;
                            res.cudaerr = cudaStatus;
                        }
                        if (!res.errstep && NwStat::success != (res.stat = setOrVerifyResult(res, compareData)))
                        {
                            res.errstep = 4;
                            compareData.calcErrors++;
                        }

                        // if the result is successful, print a dot, otherwise an x
                        if (res.stat == NwStat::success)
                        {
                            std::cout << '.' << std::flush;
                        }
                        else
                        {
                            std::cout << res.errstep << std::flush;
                        }

                        // clear cuda non-sticky errors and get possible cuda sticky errors
                        // note: repeat twice, since sticky errors cannot be cleared
                        cudaStatus = cudaGetLastError();
                        cudaStatus = cudaGetLastError();
                        if (cudaStatus != cudaSuccess)
                        {
                            std::cerr << "ERR - corrupted cuda context";
                            exit(-1);
                        }

                        nw.resetAllocsBenchmarkCycle();
                    }

                    // add the result to the results list
                    resData.resList.push_back(combineResults(resList));
                    NwResult &res = resData.resList.back();
                    // reset the multiple repetition list
                    resList.clear();

                    // print the result as a csv line to the csv output file
                    to_csv(ofsRes, res);
                    ofsRes << '\n';
                    ofsRes.flush();
                }

                // reset the algorithm parameters
                alg.alignPr().reset();

                // seqX-seqY comparison separator
                std::cout << '|' << std::flush;
            }
        }

        // algorithm separator
        std::cout << std::endl
                  << std::endl;
    }

    nw.resetAllocsBenchmarkEnd();

    // print the number of calculation errors
    if (compareData.calcErrors > 0)
    {
        std::cerr << "ERR - " << compareData.calcErrors << " calculation error(s)";
        exit(-1);
    }

    exit(0);
}
