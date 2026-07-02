export const meta = {
  name: 'qwisp-env-matrix-bench',
  description: 'Run 3-axis bench across 7 environment configs (C×SSD×mode), sequential for GPU exclusivity',
  phases: [
    { title: 'Measure', detail: '7 configs sequential: bench.sh per config, parse 3-axis cells' },
    { title: 'Audit', detail: 'completeness + anomaly cross-check' },
  ],
}

const CONFIGS = [
  { tag: 'E1-C64-fast',   c: 64,  thr: 0,     methods: 'suffix-spec bolt', label: '8GB fast-SSD' },
  { tag: 'E2-C64-slow',   c: 64,  thr: 1.5,   methods: 'suffix-spec bolt', label: '8GB slow-NAND (Neo)' },
  { tag: 'E3-C128-fast',  c: 128, thr: 0,     methods: 'suffix-spec bolt', label: '16GB fast-SSD' },
  { tag: 'E4-C128-slow',  c: 128, thr: 1.5,   methods: 'suffix-spec bolt', label: '16GB slow-NAND' },
  { tag: 'E5-C192-fast',  c: 192, thr: 0,     methods: 'suffix-spec bolt', label: '24GB fast-SSD' },
  { tag: 'E6-C192-slow',  c: 192, thr: 1.5,   methods: 'suffix-spec bolt', label: '24GB slow-NAND' },
  { tag: 'E7-C256-resident', c: 256, thr: 0,  methods: 'suffix-spec',      label: '32GB+ resident' },
]

const CELL_SCHEMA = {
  type: 'object',
  required: ['tag', 'ok', 'cells', 'aggregates', 'raw_footer', 'anomalies', 'wall_seconds'],
  properties: {
    tag: { type: 'string' },
    ok: { type: 'boolean', description: 'true if bench completed and all expected cells parsed' },
    cells: {
      type: 'array',
      items: {
        type: 'object',
        required: ['method', 'regime', 'tokps', 'fidelity_pct', 'correctness'],
        properties: {
          method: { type: 'string' },
          regime: { type: 'string' },
          tokps: { type: ['number', 'null'] },
          fidelity_pct: { type: ['number', 'null'] },
          correctness: { type: 'string', description: 'verbatim correctness string from the harness output' },
        },
      },
    },
    aggregates: {
      type: 'array',
      items: {
        type: 'object',
        required: ['method', 'mean_tokps', 'mean_fidelity_pct'],
        properties: {
          method: { type: 'string' },
          mean_tokps: { type: ['number', 'null'] },
          mean_fidelity_pct: { type: ['number', 'null'] },
        },
      },
    },
    raw_footer: { type: 'string', description: 'any skip lines, errors, or notes from the run output' },
    anomalies: { type: 'string', description: 'anything odd: NA values, checker errors, unexpected slowness, skips. Empty string if none.' },
    wall_seconds: { type: ['number', 'null'] },
  },
}

phase('Measure')
const results = []
for (const cfg of CONFIGS) {
  const nCells = cfg.methods.split(' ').length * 4
  log(`▶ ${cfg.tag} (${cfg.label}): C=${cfg.c} throttle=${cfg.thr}GB/s methods=[${cfg.methods}] — expecting ${nCells} cells`)
  const r = await agent(
    `You are running ONE benchmark configuration on this machine (repo: /Users/penta2himajin/repos/qwisp, branch main). GPU is exclusive — do NOT run any other heavy process, and run the bench exactly once.

Config: ${cfg.tag} = ${cfg.label} (cache slots C=${cfg.c}, SSD throttle ${cfg.thr} GB/s where 0 means fast-SSD unthrottled, methods: ${cfg.methods}).

Steps:
1. Run this exact command with Bash using run_in_background=true (it can take 10-30 minutes; do NOT use a foreground call, and do NOT kill it for being slow):
   /Users/penta2himajin/repos/qwisp/qwisp/bench.sh ${cfg.c} 128 ${cfg.thr} "${cfg.methods}" > /tmp/qwisp-bench-${cfg.tag}.log 2>&1; echo "EXIT=$?" >> /tmp/qwisp-bench-${cfg.tag}.log
   Record the start time via the shell (date +%s) BEFORE launching, and after completion compute wall_seconds from date +%s again.
2. While waiting, do nothing else heavy. When the background command completes you will be notified; then Read /tmp/qwisp-bench-${cfg.tag}.log in full.
3. Parse EVERY result line. The table lines look like:
     <method> <regime> <tok/s> <fidelity%> <correctness...>
   methods are literally "suffix-spec" (= strict mode) and/or "bolt"; regimes are code, agentic, longctx, shortnl. Expected cells: ${nCells}. The correctness column is free text — copy it VERBATIM and COMPLETE into the cell's correctness field (no truncation, no paraphrase). If tok/s or fidelity shows NA, use null and note it in anomalies.
4. Also parse the "equal-weight aggregate" lines into aggregates, and copy any "(skip ...)" lines, ERROR lines, or other unexpected output into raw_footer.
5. If the log shows ERROR (binary/refs missing) or EXIT= is nonzero, set ok=false and put the full error text in raw_footer. Do NOT retry more than once.
6. Sanity checks to report in anomalies (do not fail on them): strict (suffix-spec) fidelity should be ~100% at all C; fidelity and correctness at the same C should not depend on throttle; if a cell looks wildly off (e.g. strict fidelity <99%, or 0 tok/s), say so explicitly.

Return the structured result. tag="${cfg.tag}".`,
    { label: `bench:${cfg.tag}`, phase: 'Measure', schema: CELL_SCHEMA, effort: 'low' }
  )
  if (r) {
    results.push(r)
    const agg = (r.aggregates || []).map(a => `${a.method} ${a.mean_tokps} tok/s / ${a.mean_fidelity_pct}%`).join(' | ')
    log(`✔ ${cfg.tag}: ${r.cells?.length ?? 0}/${nCells} cells, agg: ${agg}${r.anomalies ? ' ⚠ ' + r.anomalies.slice(0, 160) : ''}`)
  } else {
    results.push({ tag: cfg.tag, ok: false, cells: [], aggregates: [], raw_footer: 'agent returned null (skipped or died)', anomalies: 'no data', wall_seconds: null })
    log(`✘ ${cfg.tag}: agent returned no data`)
  }
}

phase('Audit')
const audit = await agent(
  `You are auditing benchmark results for completeness and consistency. Here is the full result set as JSON:

${'```'}json
__RESULTS__
${'```'}

Expected matrix: E1-E6 are streaming configs (C=64/128/192 × fast/slow) each with methods suffix-spec+bolt × 4 regimes (code, agentic, longctx, shortnl) = 8 cells; E7 is C=256 resident with suffix-spec only = 4 cells. Total 52 cells.

Check WITHOUT running any benchmarks (you may Read log files under /tmp/qwisp-bench-*.log to resolve discrepancies):
1. Completeness: every expected cell present with non-null tok/s and fidelity? List every missing/null cell.
2. Consistency: (a) strict fidelity ≈100% everywhere? (b) at equal C, fidelity and correctness identical across fast vs slow (throttle should only change speed)? (c) speed ordering sane: fast ≥ slow at same C/method; bolt ≥ strict on streaming configs; C=192 fast strict should be in the same ballpark as C=256 resident? (d) known reference points: prior measurements had strict fast-SSD C=64≈46, C=128≈150, C=192≈168 tok/s and bolt≈166/203/208 on a canonical prompt — large deviations are worth flagging but may be prompt-mix differences, not errors.
3. Anomalies: aggregate all per-config anomaly notes; separate real issues from expected behavior.
Return a concise structured audit.`,
  {
    label: 'audit:matrix', phase: 'Audit',
    schema: {
      type: 'object',
      required: ['complete', 'missing_cells', 'consistency_findings', 'real_issues', 'expected_behaviors'],
      properties: {
        complete: { type: 'boolean' },
        missing_cells: { type: 'array', items: { type: 'string' } },
        consistency_findings: { type: 'array', items: { type: 'string' } },
        real_issues: { type: 'array', items: { type: 'string' } },
        expected_behaviors: { type: 'array', items: { type: 'string' } },
      },
    },
  }
).catch(e => ({ complete: false, missing_cells: [], consistency_findings: [], real_issues: ['audit agent failed: ' + e], expected_behaviors: [] }))

return { results, audit }