export const meta = {
  name: 'qwisp-full-matrix',
  description: 'Full environment matrix: 7 configs (C×SSD×mode doctrine) sequential via bench_batch, sonnet audit',
  whenToUse: 'Campaign-end or release measurement. Streaming C<256 = strict+bolt × fast/slow; C=256 resident = strict only. args: {gen?: number}.',
  phases: [
    { title: 'Measure', detail: '7 configs sequential, one haiku agent per config', model: 'haiku' },
    { title: 'Audit', detail: 'completeness + consistency cross-check', model: 'sonnet' },
  ],
}

const REPO = '/Users/penta2himajin/repos/qwisp'
const GEN = (args && args.gen) || 128
const CONFIGS = [
  { tag: 'E1-C64-fast', c: 64, thr: 0, methods: 'suffix-spec bolt' },
  { tag: 'E2-C64-slow', c: 64, thr: 1.5, methods: 'suffix-spec bolt' },
  { tag: 'E3-C128-fast', c: 128, thr: 0, methods: 'suffix-spec bolt' },
  { tag: 'E4-C128-slow', c: 128, thr: 1.5, methods: 'suffix-spec bolt' },
  { tag: 'E5-C192-fast', c: 192, thr: 0, methods: 'suffix-spec bolt' },
  { tag: 'E6-C192-slow', c: 192, thr: 1.5, methods: 'suffix-spec bolt' },
  { tag: 'E7-C256-resident', c: 256, thr: 0, methods: 'suffix-spec' },
]
const CELL_SCHEMA = {
  type: 'object', required: ['tag', 'ok', 'cells', 'anomalies'],
  properties: { tag: { type: 'string' }, ok: { type: 'boolean' },
    cells: { type: 'array', items: { type: 'object',
      required: ['method', 'regime', 'tokps', 'fidelity_pct', 'correctness'],
      properties: { method: { type: 'string' }, regime: { type: 'string' },
        tokps: { type: ['number', 'null'] }, fidelity_pct: { type: ['number', 'null'] },
        correctness: { type: 'string' } } } },
    anomalies: { type: 'string' } },
}

phase('Measure')
const results = []
for (const cfg of CONFIGS) {
  log(`▶ ${cfg.tag}`)
  const r = await agent(
    `Run EXACTLY once with Bash run_in_background=true (GPU exclusive, 5-25 min; slow configs use QWISP_THROTTLE_DEFER=1 to skip throttled load — steady-state tok/s unaffected):
  ${cfg.thr > 0 ? 'QWISP_THROTTLE_DEFER=1 ' : ''}${REPO}/qwisp/bench_batch.sh ${cfg.c} ${GEN} ${cfg.thr} "${cfg.methods}" > /tmp/qwisp-fm-${cfg.tag}.log 2>&1; echo "EXIT=$?" >> /tmp/qwisp-fm-${cfg.tag}.log
On completion Read the log; parse ALL table rows (tok/s, fidelity%, correctness VERBATIM+complete). ok=false + quote errors if EXIT nonzero or rows missing. tag="${cfg.tag}".`,
    { label: `bench:${cfg.tag}`, phase: 'Measure', schema: CELL_SCHEMA, model: 'haiku', effort: 'low' }
  )
  results.push(r || { tag: cfg.tag, ok: false, cells: [], anomalies: 'agent returned null' })
  if (r) log(`✔ ${cfg.tag}: ${(r.cells || []).length} cells`)
}

phase('Audit')
const audit = await agent(
  `Audit this full environment matrix for completeness and consistency:
${JSON.stringify(results, null, 1)}
Checks: (1) every expected cell present (E1-E6: methods×4 regimes; E7: strict×4)? (2) strict fidelity 100.0 everywhere (hard gate)? (3) fidelity identical across throttle at same C (determinism)? (4) fast>=slow speed sanity for strict; bolt ~throttle-flat (io=0)? (5) list all correctness FAILs verbatim. Return concise structured verdict.`,
  { label: 'audit:matrix', phase: 'Audit',
    schema: { type: 'object', required: ['complete', 'gate_pass', 'findings'],
      properties: { complete: { type: 'boolean' }, gate_pass: { type: 'boolean' },
        findings: { type: 'array', items: { type: 'string' } } } },
    model: 'sonnet' }
)

return { results, audit }
