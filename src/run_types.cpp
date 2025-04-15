#include "run_types.hpp"

////// NwRange //////

bool operator==(const NwRange& l, const NwRange& r)
{
    bool res = l.l == r.l && l.r == r.r;
    return res;
}
bool operator!=(const NwRange& l, const NwRange& r)
{
    bool res = l.l != r.l || l.r != r.r;
    return res;
}

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
NwAlgParams::NwAlgParams(Dict<std::string, NwAlgParam> params)
{
    _params = params;
    // Always allow the inital iteration, even if there are no params.
    _isEnd = false;
}

NwAlgParam& NwAlgParams::at(const std::string name)
{
    return _params.at(name);
}

const NwAlgParam& NwAlgParams::at(const std::string name) const
{
    return _params.at(name);
}

bool NwAlgParams::hasCurr() const
{
    return !_isEnd;
}
// Updates last parameter, then on iteration loop second-to-last, etc.
void NwAlgParams::next()
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

Dict<std::string, int> NwAlgParams::copy() const
{
    Dict<std::string, int> res;
    for (const auto& paramTuple : _params)
    {
        const std::string& paramName = paramTuple.first;
        int paramValue = paramTuple.second.curr();

        res[paramName] = paramValue;
    }

    return res;
}

////// NwAlgInput //////

void NwAlgInput::resetAllocsBenchmarkCycle()
{
    // First free device memory, since there is usually less of it than ram.
    // subst_gpu.clear();
    seqX_gpu.clear();
    seqY_gpu.clear();
    score_gpu.clear();
    tileHrowMat_gpu.clear();
    tileHcolMat_gpu.clear();

    // subst.clear();
    // seqX.clear();
    // seqY.clear();
    score.clear();
    tileHrowMat.clear();
    tileHcolMat.clear();

    trace.clear();
    trace.shrink_to_fit();
    tile.clear();
    tile.shrink_to_fit();
    currRow.clear();
    currRow.shrink_to_fit();
    prevRow.clear();
    prevRow.shrink_to_fit();
}

void NwAlgInput::resetAllocsBenchmarkEnd()
{
    // First free device memory, since there is usually less of it than ram.
    subst_gpu.clear();

    subst.clear();
    subst.shrink_to_fit();
    seqX.clear();
    seqX.shrink_to_fit();
    seqY.clear();
    seqY.shrink_to_fit();
}
