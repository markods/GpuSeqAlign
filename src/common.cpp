#include "common.hpp"

////// NwParam //////

NwParam::NwParam() = default;
NwParam::NwParam(std::vector<int> values)
{
    _values = values;
    _currIdx = 0;
}

int NwParam::curr() const
{
    return _values[_currIdx];
}
bool NwParam::hasCurr() const
{
    return _currIdx < _values.size();
}
void NwParam::next()
{
    _currIdx++;
}
void NwParam::reset()
{
    _currIdx = 0;
}

////// NwParams //////

NwParams::NwParams()
{
    _params = {};
    _isEnd = false;
}
NwParams::NwParams(std::map<std::string, NwParam> params)
{
    _params = params;
    // always allow the inital iteration, even if there are no params
    _isEnd = false;
}

NwParam& NwParams::operator[](const std::string name)
{
    return _params.at(name);
}

bool NwParams::hasCurr() const
{
    return !_isEnd;
}
void NwParams::next() // updates starting from the last parameter and so on
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
void NwParams::reset()
{
    for (auto iter = _params.rbegin(); iter != _params.rend(); iter++)
    {
        auto& param = iter->second;
        param.reset();
    }
    _isEnd = false;
}

std::map<std::string, int> NwParams::copy() const
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

////// NwInput //////

// free all memory allocated by the Needleman-Wunsch algorithms
void NwInput::resetAllocsBenchmarkCycle()
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
void NwInput::resetAllocsBenchmarkEnd()
{
    // NOTE: first free device memory, since there is less of it for other algorithms

    ////// device specific memory
    subst_gpu.clear();

    ////// host specific memory
    subst.clear();
    seqX.clear();
    seqY.clear();
}
