#include "file_formats.hpp"
#include "defer.hpp"
#include "fmt_guard.hpp"
#include <cctype>
#include <functional>
#include <sstream>

static void updateColIdx(std::istringstream& issLine, int64_t& iCol)
{
    if (issLine.peek() != EOF)
    {
        iCol = (int64_t)issLine.tellg();
    }
}

static NwStat error_if(
    const bool error_if_true,
    const char* message,
    const std::string& path,
    const int64_t iLine,
    const int64_t iCol,
    const NwStat returnStat,
    std::string& error_msg)
{
    if (error_if_true)
    {
        // Lines and columns start from 1.
        error_msg = path + ":" + std::to_string(1 + iLine) + ":" + std::to_string(1 + iCol) + ": " + message;
        return returnStat;
    }
    return NwStat::success;
}

static NwStat appendSequenceLine(
    std::istringstream& issLine,
    const std::string& path,
    const int64_t iLine,
    int64_t iCol,
    const Dict<std::string, int>& letterMap,
    std::vector<int>& seq,
    std::string& error_msg)
{
    if (seq.size() == 0)
    {
        // Add header element.
        seq.push_back(0);
    }

    for (int letter; (letter = issLine.peek()) != EOF; iCol++)
    {
        letter = issLine.get();
        std::string letter_str {(char)letter};
        if (!letterMap.contains(letter_str))
        {
            if (std::isspace(letter))
            {
                continue;
            }

            ZIG_TRY(NwStat::success, error_if(true /*always*/, "letter not found in substitution letters", path, iLine, iCol, NwStat::errorInvalidValue, error_msg));
        }

        auto val = letterMap.at(letter_str);
        seq.push_back(val);
    }

    ZIG_TRY(NwStat::success, error_if(issLine.fail(), "could not read letter", path, iLine, iCol, NwStat::errorInvalidValue, error_msg));
    return NwStat::success;
}

enum class FastaState
{
    expect_header,
    expect_sequence_line,
    expect_seq_line_or_header,
    eof
};

static NwStat readFastaHeaderLine(
    std::istringstream& issLine,
    const std::string& path,
    const int64_t iLine,
    int64_t iCol,
    const Dict<std::string, NwSeq>& seqMap,
    NwSeq& nwSeq,
    std::string& error_msg)
{
    // Consume '>' symbol.
    ZIG_TRY(NwStat::success, error_if(issLine.peek() != '>', "expected sequence header (>)", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));
    issLine.get();
    ZIG_TRY(NwStat::success, error_if(issLine.fail(), "expected sequence id after '>' symbol", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));
    updateColIdx(issLine, iCol);

    // Consume possible whitespace.
    issLine >> std::ws;
    updateColIdx(issLine, iCol);

    // Consume sequence id.
    issLine >> nwSeq.id;
    ZIG_TRY(NwStat::success, error_if(issLine.fail() || nwSeq.id.empty(), "expected sequence id after '>' symbol", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));
    ZIG_TRY(NwStat::success, error_if(seqMap.contains(nwSeq.id), "duplicate sequence id", path, iLine, iCol, NwStat::errorInvalidValue, error_msg));
    updateColIdx(issLine, iCol);

    // Consume possible whitespace.
    issLine >> std::ws;
    updateColIdx(issLine, iCol);

    if (issLine.peek() == EOF)
    {
        return NwStat::success;
    }

    // Consume possible line info.
    std::getline(issLine, nwSeq.info, '\n');
    ZIG_TRY(NwStat::success, error_if(issLine.fail() || nwSeq.info.empty(), "expected sequence info after sequence id", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));
    updateColIdx(issLine, iCol);

    // Trim whitespace from line info end.
    auto firstWsIter = std::find_if(nwSeq.info.rbegin(), nwSeq.info.rend(), [](char c)
    {
        return !std::isspace(c);
    }).base();
    nwSeq.info.erase(firstWsIter, nwSeq.info.end());

    return NwStat::success;
}

static NwStat readFastaSequenceLine(
    std::istringstream& issLine,
    const std::string& path,
    const int64_t iLine,
    int64_t iCol,
    const Dict<std::string, int>& letterMap,
    NwSeq& nw_seq,
    std::string& error_msg)
{
    ZIG_TRY(NwStat::success, error_if(issLine.peek() == '>', "expected sequence after header", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));
    ZIG_TRY(NwStat::success, appendSequenceLine(issLine, path, iLine, iCol, letterMap, nw_seq.seq, error_msg));

    return NwStat::success;
}

NwStat readFromFastaFormat(
    const std::string& path,
    std::istream& is,
    NwSeqData& seqData,
    const Dict<std::string, int>& letterMap,
    std::string& error_msg)
{
    FastaState state = FastaState::expect_header;
    NwSeq nw_seq {};
    std::string strLine {};
    int64_t iLine {-1};
    std::istringstream issLine {};
    int64_t iCol {};
    bool read_next_line {true};

    while (state != FastaState::eof)
    {
        if (read_next_line)
        {
            if (is.peek() != EOF)
            {
                iCol = 0;
                iLine++;

                std::getline(is, strLine, '\n');
                ZIG_TRY(NwStat::success, error_if(is.fail(), "could not read line", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));

                issLine = std::istringstream {strLine};
                issLine >> std::ws;
                updateColIdx(issLine, iCol);

                if (issLine.peek() == EOF)
                {
                    // Skip empty lines.
                    continue;
                }
            }
            else
            {
                state = FastaState::eof;
            }
        }

        switch (state)
        {
        case FastaState::expect_header:
        {
            ZIG_TRY(NwStat::success, readFastaHeaderLine(issLine, path, iLine, iCol, seqData.seqMap, nw_seq, error_msg));
            state = FastaState::expect_sequence_line;
            read_next_line = true;
            break;
        }
        case FastaState::expect_sequence_line:
        {
            ZIG_TRY(NwStat::success, readFastaSequenceLine(issLine, path, iLine, iCol, letterMap, nw_seq, error_msg));
            state = FastaState::expect_seq_line_or_header;
            read_next_line = true;
            break;
        }
        case FastaState::expect_seq_line_or_header:
        {
            if (issLine.peek() == '>')
            {
                seqData.seqMap[nw_seq.id] = nw_seq;
                nw_seq = {};

                state = FastaState::expect_header;
                read_next_line = false;
            }
            else if (issLine.peek() != EOF)
            {
                state = FastaState::expect_sequence_line;
                read_next_line = false;
            }
            else
            {
                read_next_line = true;
            }
            break;
        }
        case FastaState::eof:
        {
            if (!nw_seq.id.empty() && !nw_seq.seq.empty())
            {
                seqData.seqMap[nw_seq.id] = nw_seq;
                nw_seq = {};
            }
        }
        }
    }

    ZIG_TRY(NwStat::success, error_if(is.bad() || (is.fail() && !is.eof()), "file truncated", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));
    ZIG_TRY(NwStat::success, error_if(state == FastaState::expect_header, "expected sequence header (>)", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));
    ZIG_TRY(NwStat::success, error_if(state == FastaState::expect_sequence_line, "expected sequence after header", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));

    return NwStat::success;
}

static NwStat readSeqIdAndRange(
    std::istringstream& issLine,
    const std::string& path,
    const int64_t iLine,
    int64_t iCol,
    std::string& seqId,
    NwRange& seqRange,
    const Dict<std::string, NwSeq>& seqMap,
    std::string& error_msg)
{
    // Consume possible whitespace.
    issLine >> std::ws;
    updateColIdx(issLine, iCol);

    // Consume sequence id.
    seqId = "";
    for (int c; (c = issLine.peek()) != EOF; issLine.get())
    {
        if (std::isspace(c) || c == '[')
        {
            break;
        }
        seqId.push_back((char)c);
    }
    ZIG_TRY(NwStat::success, error_if(issLine.fail() || seqId.empty(), "expected sequence id", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));
    ZIG_TRY(NwStat::success, error_if(!seqMap.contains(seqId), "unknown sequence id", path, iLine, iCol, NwStat::errorInvalidValue, error_msg));
    updateColIdx(issLine, iCol);

    int64_t seqSizeNoHeader = (int64_t)seqMap.at(seqId).seq.size() - 1 /*without header*/;
    seqRange.l = 0;
    seqRange.r = seqSizeNoHeader;
    seqRange.lNotDefault = false;
    seqRange.rNotDefault = false;

    if (issLine.peek() != '[')
    {
        // Default range (whole sequence).
        return NwStat::success;
    }

    // Consume left angle bracket '['.
    issLine.get();
    ZIG_TRY(NwStat::success, error_if(issLine.fail(), "could not parse '['", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));
    updateColIdx(issLine, iCol);

    // Consume possible whitespace.
    issLine >> std::ws;
    updateColIdx(issLine, iCol);

    // Consume possible left range index.
    if (auto c = issLine.peek(); c != ':')
    {
        ZIG_TRY(NwStat::success, error_if(!std::isdigit(c) && c != '+' && c != '-', "expected a number", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));
        issLine >> seqRange.l;
        seqRange.lNotDefault = true;

        ZIG_TRY(NwStat::success, error_if(issLine.fail(), "expected a number", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));
        ZIG_TRY(NwStat::success, error_if(seqRange.l < 0, "left bound must be non-negative", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));
        ZIG_TRY(NwStat::success, error_if(seqRange.l >= seqSizeNoHeader, "left bound greater than or equal to sequence length", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));
        updateColIdx(issLine, iCol);
    }

    // Consume possible whitespace.
    issLine >> std::ws;
    updateColIdx(issLine, iCol);

    // Consume colon.
    ZIG_TRY(NwStat::success, error_if(issLine.peek() != ':', "expected ':'", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));
    issLine.get();
    ZIG_TRY(NwStat::success, error_if(issLine.fail(), "could not parse ':'", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));
    updateColIdx(issLine, iCol);

    // Consume possible whitespace.
    issLine >> std::ws;
    updateColIdx(issLine, iCol);

    // Consume possible right range index.
    if (auto c = issLine.peek(); c != ']')
    {
        ZIG_TRY(NwStat::success, error_if(!std::isdigit(c) && c != '+' && c != '-', "expected a number", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));
        issLine >> seqRange.r;
        seqRange.rNotDefault = true;

        ZIG_TRY(NwStat::success, error_if(issLine.fail(), "expected a number", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));
        ZIG_TRY(NwStat::success, error_if(seqRange.r <= seqRange.l, "right bound must be greater than left", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));
        ZIG_TRY(NwStat::success, error_if(seqRange.r > seqSizeNoHeader, "right bound greater than sequence length", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));
        updateColIdx(issLine, iCol);
    }

    // Consume possible whitespace.
    issLine >> std::ws;
    updateColIdx(issLine, iCol);

    // Consume right angle bracket (']').
    ZIG_TRY(NwStat::success, error_if(issLine.peek() != ']', "expected ']'", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));
    issLine.get();
    ZIG_TRY(NwStat::success, error_if(issLine.fail(), "could not parse ']'", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));
    updateColIdx(issLine, iCol);

    return NwStat::success;
}

NwStat readFromSeqPairFormat(
    const std::string& path,
    std::istream& is,
    NwSeqPairData& seqPairData,
    const Dict<std::string, NwSeq>& seqMap,
    std::string& error_msg)
{
    NwSeqPair seqPair {};
    std::string strLine {};
    std::istringstream issLine {};
    int64_t iLine {-1};
    int64_t iCol {0};

    while (true)
    {
        if (is.peek() != EOF)
        {
            iLine++;
            iCol = 0;

            std::getline(is, strLine, '\n');
            ZIG_TRY(NwStat::success, error_if(is.fail(), "could not read line", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));

            issLine = std::istringstream {strLine};
            issLine >> std::ws;
            updateColIdx(issLine, iCol);

            if (issLine.peek() == EOF)
            {
                // Skip empty lines.
                continue;
            }
        }
        else
        {
            break;
        }

        ZIG_TRY(NwStat::success, readSeqIdAndRange(issLine, path, iLine, iCol, seqPair.seqY_id, seqPair.seqY_range, seqMap, error_msg));
        ZIG_TRY(NwStat::success, readSeqIdAndRange(issLine, path, iLine, iCol, seqPair.seqX_id, seqPair.seqX_range, seqMap, error_msg));
        issLine >> std::ws;
        updateColIdx(issLine, iCol);

        if (issLine.peek() != EOF)
        {
            ZIG_TRY(NwStat::success, error_if(true, "expected next line", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));
        }

        seqPairData.pairList.push_back(seqPair);
        seqPair = {};
    }

    ZIG_TRY(NwStat::success, error_if(is.bad() || (is.fail() && !is.eof()), "file truncated", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));
    ZIG_TRY(NwStat::success, error_if(seqPairData.pairList.size() == 0, "expected at least one sequence pair", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));

    return NwStat::success;
}

template <typename T>
static void fieldToTsv(
    std::ostream& os,
    const char* const fieldName,
    const T fieldVal,
    const TsvPrintCtl& printCtl,
    std::optional<std::function<void(std::ostream&)>> format_fn = std::nullopt,
    const bool writeSep = true)
{
    if (writeSep)
    {
        os << '\t';
    }
    if (printCtl.writeColName)
    {
        os << fieldName;
    }
    if (printCtl.writeValue)
    {
        if (format_fn.has_value())
        {
            FormatFlagsGuard fg {os};
            format_fn.value()(os);
            os << fieldVal;
        }
        else
        {
            os << fieldVal;
        }
    }
}

static std::string seqIdAndRangeToString(const std::string seq_id, const NwRange& seq_range)
{
    std::string res {seq_id};

    if (seq_range.lNotDefault || seq_range.rNotDefault)
    {
        res.append("[");
        if (seq_range.lNotDefault)
        {
            res.append(std::to_string(seq_range.l));
        }
        res.append(":");
        if (seq_range.rNotDefault)
        {
            res.append(std::to_string(seq_range.r));
        }
        res.append("]");
    }

    return res;
}

NwStat writeNwResultToTsv(std::ostream& os, const NwAlgResult& res, const TsvPrintCtl printCtl)
{
    if (printCtl.writeColName == printCtl.writeValue)
    {
        return NwStat::errorInvalidValue;
    }

    auto fmt1 = [](std::ostream& os)
    { os << std::hex << std::setw(8) << std::setfill('0'); };
    auto fmt2 = [](std::ostream& os)
    { os << std::fixed << std::setprecision(4); };

    fieldToTsv(os, "alg_name", res.algName, printCtl, std::nullopt, false /*writeSep*/);
    fieldToTsv(os, "seqY_idx", res.seqY_idx, printCtl);
    fieldToTsv(os, "seqX_idx", res.seqX_idx, printCtl);
    fieldToTsv(os, "seqY_id", seqIdAndRangeToString(res.seqY_id, res.seqY_range), printCtl);
    fieldToTsv(os, "seqX_id", seqIdAndRangeToString(res.seqX_id, res.seqX_range), printCtl);

    fieldToTsv(os, "seqY_len", res.seqY_len, printCtl);
    fieldToTsv(os, "seqX_len", res.seqX_len, printCtl);
    fieldToTsv(os, "subst_name", res.substName, printCtl);
    fieldToTsv(os, "gapo_cost", res.gapoCost, printCtl);
    fieldToTsv(os, "warmup_runs", res.warmup_runs, printCtl);
    fieldToTsv(os, "sample_runs", res.sample_runs, printCtl);
    fieldToTsv(os, "last_run_idx", res.last_run_idx, printCtl);

    nlohmann::ordered_json algParamsJson = res.algParams;
    fieldToTsv(os, "alg_params", algParamsJson.dump(), printCtl);

    fieldToTsv(os, "err_step", res.errstep, printCtl);
    fieldToTsv(os, "nw_stat", int(res.stat), printCtl);
    fieldToTsv(os, "cuda_stat", int(res.cudaStat), printCtl);

    fieldToTsv(os, "align_cost", res.align_cost, printCtl);
    if (printCtl.fPrintScoreStats)
    {
        fieldToTsv(os, "score_hash", res.score_hash, printCtl, fmt1);
    }
    if (printCtl.fPrintTraceStats)
    {
        fieldToTsv(os, "trace_hash", res.trace_hash, printCtl, fmt1);
    }

    fieldToTsv(os, "host_allocs", res.hostAllocations, printCtl);
    fieldToTsv(os, "device_allocs", res.deviceAllocations, printCtl);

    fieldToTsv(os, "align.alloc", res.sw_align.get_or_default("align.alloc"), printCtl, fmt2);
    fieldToTsv(os, "align.cpy_dev", res.sw_align.get_or_default("align.cpy_dev"), printCtl, fmt2);
    fieldToTsv(os, "align.init_hdr", res.sw_align.get_or_default("align.init_hdr"), printCtl, fmt2);
    fieldToTsv(os, "align.calc_init", res.sw_align.get_or_default("align.calc_init"), printCtl, fmt2);
    fieldToTsv(os, "align.calc", res.sw_align.get_or_default("align.calc"), printCtl, fmt2);
    fieldToTsv(os, "align.cpy_host", res.sw_align.get_or_default("align.cpy_host"), printCtl, fmt2);
    if (printCtl.fPrintScoreStats)
    {
        fieldToTsv(os, "hash.calc", res.sw_hash.get_or_default("hash.calc"), printCtl, fmt2);
    }
    if (printCtl.fPrintTraceStats)
    {
        fieldToTsv(os, "trace.alloc", res.sw_trace.get_or_default("trace.alloc"), printCtl, fmt2);
        fieldToTsv(os, "trace.calc", res.sw_trace.get_or_default("trace.calc"), printCtl, fmt2);
        fieldToTsv(os, "edit_trace", res.edit_trace, printCtl);
    }

    os << '\n';
    return NwStat::success;
}
