#include "run_types.hpp"

////// NwAlgParam //////

NwAlgParam::NwAlgParam() = default;
NwAlgParam::NwAlgParam(std::vector<int> values)
{
    _values = values;
    _currIdx = 0;
}

int NwAlgParam::curr() const
{
    return _values[_currIdx];
}
bool NwAlgParam::hasCurr() const
{
    return _currIdx < _values.size();
}
void NwAlgParam::next()
{
    _currIdx++;
}
void NwAlgParam::reset()
{
    _currIdx = 0;
}

////// NwAlgParams //////

NwAlgParams::NwAlgParams()
{
    _params = {};
    _isEnd = false;
}
NwAlgParams::NwAlgParams(std::map<std::string, NwAlgParam> params)
{
    _params = params;
    // always allow the inital iteration, even if there are no params
    _isEnd = false;
}

NwAlgParam& NwAlgParams::operator[](const std::string name)
{
    return _params.at(name);
}

bool NwAlgParams::hasCurr() const
{
    return !_isEnd;
}
void NwAlgParams::next() // updates starting from the last parameter and so on
{
    for (auto iter = _params.rbegin(); iter != _params.rend(); iter++)
    {
        auto& param = iter->second;
        param.next();

        if (param.hasCurr())
        {
            return;
        }
        param.reset();
    }
    _isEnd = true;
}
void NwAlgParams::reset()
{
    for (auto iter = _params.rbegin(); iter != _params.rend(); iter++)
    {
        auto& param = iter->second;
        param.reset();
    }
    _isEnd = false;
}

std::map<std::string, int> NwAlgParams::copy() const
{
    std::map<std::string, int> res;
    for (const auto& paramTuple : _params)
    {
        const std::string& paramName = paramTuple.first;
        int paramValue = paramTuple.second.curr();

        res[paramName] = paramValue;
    }

    return res;
}

////// NwAlgInput //////

// free all memory allocated by the Needleman-Wunsch algorithms
void NwAlgInput::resetAllocsBenchmarkCycle()
{
    // NOTE: first free device memory, since there is less of it for other algorithms

    ////// device specific memory
    // subst_gpu.clear();
    seqX_gpu.clear();
    seqY_gpu.clear();
    score_gpu.clear();
    ////// sparse representation of score matrix
    tileHrowMat_gpu.clear();
    tileHcolMat_gpu.clear();

    ////// host specific memory
    // subst.clear();
    // seqX.clear();
    // seqY.clear();
    score.clear();
    ////// sparse representation of score matrix
    tileHrowMat.clear();
    tileHcolMat.clear();
}

// free all remaining memory not cleared by resetAllocs
void NwAlgInput::resetAllocsBenchmarkEnd()
{
    // NOTE: first free device memory, since there is less of it for other algorithms

    ////// device specific memory
    subst_gpu.clear();

    ////// host specific memory
    subst.clear();
    seqX.clear();
    seqY.clear();
}
