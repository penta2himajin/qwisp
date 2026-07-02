export const meta = {
  name: 'qwisp-regression-triage',
  description: 'Lossless-regression discriminator playbook: token compare, cross-C, OFDBG, shape/op probes (evidence only — Fable diagnoses)',
  whenToUse: 'When a strict cell reads fidelity <100%. args: {regime: "code|agentic|longctx|shortnl", c: number}. Produces an evidence bundle; root-cause analysis is done by the main loop, not here.',
  phases: [
    { title: 'Probe', detail: 'sequential GPU discriminators (sonnet, follows the fixed playbook)', model: 'sonnet' },
  ],
}

const REPO = '/Users/penta2himajin/repos/qwisp'
const BIN = REPO + '/swift/.xcode-build-rel/Build/Products/Release/qwisp-poc'
const MODELP = '$HOME/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16'
const PY = '$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python3'
const regime = (args && args.regime) || 'agentic'
const C = (args && args.c) || 64

phase('Probe')
const evidence = await agent(
  `You are executing a FIXED diagnostic playbook for a strict-lossless regression (regime=${regime}, C=${C}). GPU exclusive — run every command strictly sequentially, foreground Bash timeout 600000. Repo ${REPO}. Base command template:
  <ENV> QWISP_RUN=suffix-spec QWISP_DUMP_TOKENS=1 QWISP_MODEL="${MODELP}" QWISP_MTP_REF=${REPO}/refs/${regime}.safetensors QWISP_CACHE_C=${C} QWISP_GEN=128 "${BIN}" stream
Steps (record verbatim outputs for each; the 品質/fidelity line + any OFDBG stderr):
1. BASELINE+POSITION: run with QWISP_OVERFLOW_DBG=1, dump to /tmp/qt-base.log(+.err). Extract OUT_TOKENS, run: PYTHONPATH=${REPO} "${PY}" ${REPO}/qwisp/bench_refs.py --help >/dev/null 2>&1; PYTHONPATH=${REPO} "${PY}" ${REPO}/qwisp/bench_tokcmp.py ${REPO}/refs/${regime}.safetensors /tmp/qt-base.dump — record first divergence index + first5 mismatches + OFDBG event list.
2. TF CHECK: QWISP_RUN=mlx-fidelity QWISP_SKIPMODE=0 same ref/C — record the full fidelity line incl. per-position mismatches (near-tie gaps). TF<100% means REF-side suspicion (regenerate refs with canonical config: f32-full+fuseOFF+chunk8+M=1); TF=100% + free-run<100% means engine free-run drift.
3. CROSS-C: baseline run at the OTHER C tier (${C === 64 ? 256 : 64}) — same divergence => C-independent (op-level); clean => C-machinery (guard/capacity).
4. SHAPE PROBES at C=${C}: (a) QWISP_DRAFT_K=8, (b) QWISP_DRAFT_K=4, (c) QWISP_VERIFY_SEQ=0 (expect worse — sanity), (d) QWISP_FUSE_GDN=0 vs 1. Record 品質 for each.
5. If OFDBG showed overflow events: rerun with drafts capped below the overflow threshold to decorrelate guard involvement.
Do NOT modify any files. Do NOT attempt fixes or interpretation beyond one factual sentence per step. Return the complete evidence bundle.`,
  { label: `probe:${regime}-C${C}`, phase: 'Probe',
    schema: { type: 'object', required: ['steps', 'divergence_index', 'raw_lines'],
      properties: { steps: { type: 'array', items: { type: 'object',
        required: ['name', 'result'], properties: { name: { type: 'string' }, result: { type: 'string' } } } },
        divergence_index: { type: ['number', 'null'] },
        raw_lines: { type: 'array', items: { type: 'string' } } } },
    model: 'sonnet' }
)

return evidence
