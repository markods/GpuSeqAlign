#include "file_formats.hpp"
#include "fmt_guard.hpp"

void writeResultHeaderToTsv(std::ostream& os,
    bool fPrintScoreStats,
    bool fPrintTraceStats)
{
    FormatFlagsGuard fg {os};
    os.fill(' ');

    os << "alg_name";
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
