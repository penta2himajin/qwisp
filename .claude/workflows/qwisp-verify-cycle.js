export const meta = {
  name: 'qwisp-verify-cycle',
  description: 'Post-change verification gate: build + bench_batch + lossless/speed audit (no commit — Fable reviews)',
  whenToUse: 'After any engine change. args: {configs?: [{c, thr, methods}], build?: bool, baselines?: object}. Default: C=64 fast both methods + C=256 strict.',
  phases: [
    { title: 'Bench', detail: 'one haiku agent: sequential build+bench chain, parse cells', model: 'haiku' },
    { title: 'Audit', detail: 'sonnet agent: lossless gate + speed deltas vs baselines', model: 'sonnet' },
  ],
}

const REPO = '/Users/penta2himajin/repos/qwisp'
const configs = (args && args.configs) || [
  { c: 64, thr: 0, methods: 'suffix-spec bolt' },
  { c: 256, thr: 0, methods: 'suffix-spec' },
]
const doBuild = !args || args.build !== false

const CELLS_SCHEMA = {
  type: 'object',
  required: ['ok', 'runs', 'notes'],
  properties: {
    ok: { type: 'boolean' },
    runs: { type: 'array', items: { type: 'object',
      required: ['c', 'thr', 'cells'],
      properties: { c: { type: 'number' }, thr: { type: 'number' },
        cells: { type: 'array', items: { type: 'object',
          required: ['method', 'regime', 'tokps', 'fidelity_pct', 'correctness'],
          properties: { method: { type: 'string' }, regime: { type: 'string' },
            tokps: { type: ['number', 'null'] }, fidelity_pct: { type: ['number', 'null'] },
            correctness: { type: 'string' } } } } } } },
    notes: { type: 'string' },
  },
}

phase('Bench')
const chain = configs.map((k, i) =>
  `${REPO}/qwisp/bench_batch.sh ${k.c} 128 ${k.thr} "${k.methods}" > /tmp/qwisp-vc-${i}.log 2>&1; echo "EXIT=$?" >> /tmp/qwisp-vc-${i}.log`
).join('\n  ')
const bench = await agent(
  `Run this verification chain EXACTLY (one Bash call with run_in_background=true; sequential; may take 10-30 min; do NOT run anything else heavy — GPU is exclusive):
${doBuild ? `  cd ${REPO}/swift && xcodebuild build -scheme qwisp-poc -configuration Release -destination 'platform=macOS' -derivedDataPath ./.xcode-build-rel -skipPackagePluginValidation 2>&1 | tail -1\n` : ''}  ${chain}
When it completes, Read each /tmp/qwisp-vc-*.log and parse every table row (method regime tok/s fidelity% correctness-verbatim). ok=false if any EXIT nonzero or build failed (quote the error in notes). Report parse anomalies (NA cells, checker errors) in notes.`,
  { label: 'bench:chain', phase: 'Bench', schema: CELLS_SCHEMA, model: 'haiku', effort: 'low' }
)

phase('Audit')
const audit = await agent(
  `Audit these benchmark results as a strict verification gate. Results JSON:
${JSON.stringify(bench, null, 1)}
${args && args.baselines ? `Baselines for speed comparison: ${JSON.stringify(args.baselines)}` : 'No baselines provided — report absolute values only.'}
Rules:
1. LOSSLESS GATE (hard): every suffix-spec cell must have fidelity == 100.0. Any other value = GATE FAIL (name the cells).
2. bolt fidelity: report values (no hard gate; L3), flag any drop >3pt vs baseline if given.
3. correctness: list every FAIL with its verbatim string; strict FAILs are gate-relevant, bolt FAILs are delta-informational.
4. speed: compute per-cell delta vs baselines when given; flag regressions >5%.
Return a concise verdict.`,
  { label: 'audit:gate', phase: 'Audit',
    schema: { type: 'object', required: ['gate_pass', 'failures', 'speed_notes', 'summary'],
      properties: { gate_pass: { type: 'boolean' }, failures: { type: 'array', items: { type: 'string' } },
        speed_notes: { type: 'array', items: { type: 'string' } }, summary: { type: 'string' } } },
    model: 'sonnet' }
)

return { bench, audit }
