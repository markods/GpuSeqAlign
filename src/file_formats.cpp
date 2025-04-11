#include "file_formats.hpp"
#include "defer.hpp"
#include "fmt_guard.hpp"
#include <sstream>

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
        error_msg = path + ":" + std::to_string(iLine) + ":" + std::to_string(iCol) + ": " + message;
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

    for (char letter; (issLine >> letter); iCol++)
    {
        std::string letter_str {letter};
        if (!letterMap.contains(letter_str))
        {
            if (std::isspace(letter))
            {
                continue;
            }

            ZIG_TRY(NwStat::success, error_if(true /*always*/, "letter not found in substitution letters", path, iLine, iCol, NwStat::errorInvalidValue, error_msg));
        }

        // TODO: use char for letterMap
        auto val = letterMap.at(letter_str);
        seq.push_back(val);
    }

    ZIG_TRY(NwStat::success, error_if(issLine.bad(), "could not read letter", path, iLine, iCol, NwStat::errorInvalidValue, error_msg));
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
    NwSeq& nw_seq,
    std::string& error_msg)
{
    ZIG_TRY(NwStat::success, error_if(issLine.peek() != '>', "expected sequence header (>)", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));
    issLine.get(); // Consume '>' symbol.

    issLine >> nw_seq.id;
    ZIG_TRY(NwStat::success, error_if(issLine.fail() || nw_seq.id.empty(), "expected sequence id after '>' symbol", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));

    issLine >> std::ws;
    if (!issLine.eof())
    {
        iCol = (int64_t)issLine.tellg();

        std::getline(issLine, nw_seq.info, '\n');
        ZIG_TRY(NwStat::success, error_if(issLine.fail() || nw_seq.info.empty(), "expected sequence info after sequence id", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));
    }

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
    int64_t iLine {-1 + 1 /*lines start from 1, not 0*/};
    int64_t iCol {1 /*columns start from 1, not 0*/};
    bool read_next_line {true};

    while (state != FastaState::eof)
    {
        if (read_next_line)
        {
            if (!is.eof())
            {
                std::getline(is, strLine, '\n');
                iLine++;
                iCol = 0;

                ZIG_TRY(NwStat::success, error_if(is.bad(), "could not read line", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));
            }
            else
            {
                state = FastaState::eof;
            }
        }

        std::istringstream issLine {strLine};
        issLine >> std::ws;
        if (state != FastaState::eof && issLine.eof())
        {
            // Skip empty lines.
            continue;
        }

        switch (state)
        {
        case FastaState::expect_header:
        {
            ZIG_TRY(NwStat::success, readFastaHeaderLine(issLine, path, iLine, iCol, nw_seq, error_msg));
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

    ZIG_TRY(NwStat::success, error_if(state == FastaState::expect_header, "expected sequence header (>)", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));
    ZIG_TRY(NwStat::success, error_if(state == FastaState::expect_sequence_line, "expected sequence after header", path, iLine, iCol, NwStat::errorInvalidFormat, error_msg));

    return NwStat::success;
}

// TODO: combine with writeResultLineToTsv
void writeResultHeaderToTsv(std::ostream& os,
    bool fPrintScoreStats,
    bool fPrintTraceStats)
{
    FormatFlagsGuard fg {os};
    os.fill(' ');

    os << "alg_name";
    os << "\t" << "seqY_idx";
    os << "\t" << "seqX_idx";
    os << "\t" << "seqY_id";
    os << "\t" << "seqX_id";

    os << "\t" << "seqY_len";
    os << "\t" << "seqX_len";
    os << "\t" << "subst_name";
    os << "\t" << "gapo_cost";
    os << "\t" << "warmup_runs";
    os << "\t" << "sample_runs";

    os << "\t" << "alg_params";

    os << "\t" << "err_step";
    os << "\t" << "nw_stat";
    os << "\t" << "cuda_stat";

    os << "\t" << "align_cost";
    if (fPrintScoreStats)
    {
        os << "\t" << "score_hash";
    }
    if (fPrintTraceStats)
    {
        os << "\t" << "trace_hash";
    }

    os << "\t" << "align.alloc";
    os << "\t" << "align.cpy_dev";
    os << "\t" << "align.init_hdr";
    os << "\t" << "align.calc_init";
    os << "\t" << "align.calc";
    os << "\t" << "align.cpy_host";
    if (fPrintScoreStats)
    {
        os << "\t" << "hash.calc";
    }
    if (fPrintTraceStats)
    {
        os << "\t" << "trace.alloc";
        os << "\t" << "trace.calc";
    }

    os << '\n';
}

static void lapTimeToTsv(std::ostream& os, float lapTime)
{
    os << std::fixed << std::setprecision(4) << lapTime;
}

void writeResultLineToTsv(
    std::ostream& os,
    const NwAlgResult& res,
    bool fPrintScoreStats,
    bool fPrintTraceStats)
{
    FormatFlagsGuard fg {os};

    os << res.algName;
    os << "\t" << res.seqY_idx;
    os << "\t" << res.seqX_idx;
    os << "\t" << res.seqY_id;
    os << "\t" << res.seqX_id;

    os << "\t" << res.seqY_len;
    os << "\t" << res.seqX_len;
    os << "\t" << res.substName;
    os << "\t" << res.gapoCost;
    os << "\t" << res.warmup_runs;
    os << "\t" << res.sample_runs;

    nlohmann::ordered_json algParamsJson = res.algParams;
    os << "\t" << algParamsJson.dump();

    os << "\t" << res.errstep;
    os << "\t" << int(res.stat);
    os << "\t" << int(res.cudaStat);

    os << "\t" << res.align_cost;
    if (fPrintScoreStats)
    {
        os.fill('0');
        os << "\t" << std::setw(10) << res.score_hash;
        fg.restore();
    }
    if (fPrintTraceStats)
    {
        os.fill('0');
        os << "\t" << std::setw(10) << res.trace_hash;
        fg.restore();
    }

    os << "\t";
    lapTimeToTsv(os, res.sw_align.get_or_default("align.alloc"));
    os << "\t";
    lapTimeToTsv(os, res.sw_align.get_or_default("align.cpy_dev"));
    os << "\t";
    lapTimeToTsv(os, res.sw_align.get_or_default("align.init_hdr"));
    os << "\t";
    lapTimeToTsv(os, res.sw_align.get_or_default("align.calc_init"));
    os << "\t";
    lapTimeToTsv(os, res.sw_align.get_or_default("align.calc"));
    os << "\t";
    lapTimeToTsv(os, res.sw_align.get_or_default("align.cpy_host"));

    if (fPrintScoreStats)
    {
        os << "\t";
        lapTimeToTsv(os, res.sw_hash.get_or_default("hash.calc"));
    }

    if (fPrintTraceStats)
    {
        os << "\t";
        lapTimeToTsv(os, res.sw_trace.get_or_default("trace.alloc"));
        os << "\t";
        lapTimeToTsv(os, res.sw_trace.get_or_default("trace.calc"));
    }

    os << "\n";
}
