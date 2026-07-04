export const meta = {
  name: 'qwisp-fusion-loop',
  description: 'Reusable qwisp optimization loop: (phase=recon) Opus deep-dive OR (phase=loop) Sonnet locked tests → GLM-5.2 implement (Pi harness, Sonnet-driven with fallback) → Sonnet adversarial review, iterate ≤maxRounds. Steps 1 (spec) and 5 (final audit) belong to the driver (Fable) outside this workflow.',
  whenToUse: 'qwisp の最適化 wave を回すとき。args: {phase:"recon", reconPrompt, reconModel?} で事前調査、{phase:"loop", spec, lockDir, expectedTotalBefore, testBrief, implBrief?, maxRounds?} で TDD ループ。',
  phases: [
    { title: 'Recon', detail: 'phase=recon のみ: Opus 事前調査' },
    { title: 'Author tests', detail: 'phase=loop: Sonnet write-locked RED tests' },
    { title: 'Implement+Review', detail: 'GLM-5.2 (Pi) implements via Sonnet driver w/ fallback → Sonnet adversarial review, loop' },
  ],
}

const REPO = '/Users/penta2himajin/repos/qwisp'
const BIN = 'swift/.xcode-build-rel/Build/Products/Release/qwisp-poc'
const BUILD = "cd swift && xcodebuild build -scheme qwisp-poc -destination 'platform=macOS' -derivedDataPath ./.xcode-build-rel -configuration Release -skipPackagePluginValidation 2>&1 | tail -3"

const COMMON = `
Repo: ${REPO}, branch feat/raw-verify. GPU EXCLUSIVE — you are the only build/model job.
Campaign doctrine (notes/06-08): each fusion atom = ONE dispatch, production wiring is gated by paired-A/B prof (G5) not unit tests; test references = production kernels ONLY (no MLX arithmetic stand-ins); if a locked test contradicts production kernels, STOP and report; fused-kernel grids must match the widest-parallelism absorbed op; fusion wins concentrate in long dependent chains.
Build: ${BUILD}   Binary: ${BIN}   Fast gate: QWISP_RUN=raw-tests ${BIN} stream → "RAWTESTS X/X".
Prof (G5 instrument): QWISP_RUN=raw-fused-prof QWISP_PROF_AB=1 QWISP_MTP_REF=refs/code.safetensors ${BIN} stream (median primary, M=17 lane unusable). G2: QWISP_RUN=raw-spec QWISP_RAW_FUSED=1 <flags> QWISP_RAWSPEC_CHECK=1 QWISP_DUMP_TOKENS=1 QWISP_GEN=128 QWISP_MTP_REF=refs/<regime>.safetensors.
Do NOT commit (the driver is the only committer). Bit-exact only, no tolerances.
`

// ---------- args normalization(文字列で渡っても壊れない防御)----------
let A = args
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = null } }
if (!A || typeof A !== 'object' || !A.phase) {
  return { aborted: 'args missing/unparseable — expected {phase:"recon"|"loop", ...}. got: ' + String(args).slice(0, 200) }
}

// ---------- phase: recon ----------
if (A.phase === 'recon') {
  phase('Recon')
  const recon = await agent(
    `${COMMON}\nYOU ARE THE RECON INVESTIGATOR. ${A.reconPrompt}`,
    { model: A.reconModel ?? 'opus', effort: 'high', phase: 'Recon', label: 'recon' }
  )
  return { recon }
}

// ---------- phase: loop ----------
const SPEC = A.spec
const LOCKDIR = A.lockDir
if (!SPEC || !LOCKDIR) { return { aborted: 'loop mode requires args.spec and args.lockDir — refusing to run with undefined contract/lock (past incident: tests were authored against "undefined").' } }
const MAXR = A.maxRounds ?? 3
const TOTB = A.expectedTotalBefore       // RAWTESTS total before this wave

const TEST_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['newTotal', 'redConfirmed', 'stubSignatures', 'summary'],
  properties: {
    newTotal: { type: 'integer' }, redConfirmed: { type: 'boolean' },
    stubSignatures: { type: 'string' }, summary: { type: 'string' },
  },
}
const ROUND_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass', 'testIntegrityOk', 'structuralOk', 'rawtests', 'g2', 'g5', 'requiredFixes', 'concerns', 'summary'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if integrity+structural+RAWTESTS all-pass+G2 identity/LOSSLESS+G5 per spec gates' },
    testIntegrityOk: { type: 'boolean' },
    structuralOk: { type: 'boolean', description: 'production wiring verified by reading encode code (not merely unit tests)' },
    rawtests: { type: 'string' }, g2: { type: 'string' }, g5: { type: 'string' },
    requiredFixes: { type: 'array', items: { type: 'string' } },
    concerns: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string' },
  },
}

phase('Author tests')
const tests = await agent(
  `${COMMON}
YOU ARE THE TEST AUTHOR (Sonnet). Contract: ${SPEC} (read fully). Write the WRITE-LOCKED tests per its gate section, following the established idiom in swift/Sources/QwispCore/RawVerifyTests.swift (stub-RED: place nil-returning stub APIs marked "STUB — implementation pending" in the implementation file; run("name"){(Bool,String)}; bitEqual; references = production kernels only; include M-invariance and adversarial cases per spec).
${A.testBrief ?? ''}
Then: build, run raw-tests, confirm existing ${TOTB} PASS + new tests RED. WRITE-LOCK: mkdir -p ${LOCKDIR} && cp swift/Sources/QwispCore/RawVerifyTests.swift ${LOCKDIR}/
Return the schema.`,
  { model: 'sonnet', effort: 'high', phase: 'Author tests', label: 'author-tests', schema: TEST_SCHEMA }
)
log(`tests: total→${tests?.newTotal}, RED=${tests?.redConfirmed}`)

phase('Implement+Review')
let feedback = ''
let last = null
for (let round = 1; round <= MAXR; round++) {
  // Step 3: implementation — GLM-5.2 on Pi harness, driven by a Sonnet driver agent with fallback.
  await agent(
    `${COMMON}
YOU ARE THE IMPLEMENTATION DRIVER (Sonnet), round ${round}. The preferred implementer is GLM-5.2 via the Pi-harness glm-code CLI (per ~/.claude/CLAUDE.md). Contract: ${SPEC}. Stub signatures (fixed): ${tests?.stubSignatures}
${A.implBrief ?? ''}
${round === 1 ? '' : `ADVERSARIAL REVIEW FEEDBACK to fix this round:\n${feedback}\n`}
PROCEDURE:
1. Write a complete self-contained task brief to ${REPO}/.claude-glm-task.md (contract pointer, current state, per-item requirements incl. this round's review feedback, build/test commands, "no test edits, no commits, no delegation" rules). Keep the glm-code PROMPT itself short (one line pointing at the file) — long prompts trigger long silent thinking.
2. Launch GLM (bash): cd ${REPO} && GLM_DIR=${REPO} GLM_IDLE_TIMEOUT=1200 zsh -ic '~/bin/glm-code --allow "cd *" --allow "*xcodebuild*" --allow "*qwisp-poc*" ".claude-glm-task.md を読んでタスクを完遂して。"' — this blocks until GLM finishes or idle-kills; capture the tail. Progress streams to ~/.cache/glm-code/latest.log.
3. If GLM stalls/dies: check git diff for saved partial edits, resume ONCE with glm-code -c "続きを完了して。<what remains>". If the resume also fails or produces no edits: IMPLEMENT THE REMAINDER YOURSELF (Sonnet fallback — this is the standing rule).
4. Whoever implements: build + raw-tests loop until all tests pass (${tests?.newTotal ?? 'expected total'}), then ONE G2 smoke pair per the contract.
5. Report as final text: who implemented what (GLM vs fallback), RAWTESTS line, G2 smoke, dispatch/structural counts per the contract's reporting requirements.
Do NOT modify swift/Sources/QwispCore/RawVerifyTests.swift.`,
    { model: 'sonnet', effort: 'high', phase: 'Implement+Review', label: `impl:r${round}` }
  )

  // Step 4: adversarial review.
  last = await agent(
    `${COMMON}
YOU ARE THE ADVERSARIAL REVIEWER (Sonnet), round ${round}. Contract: ${SPEC} — its gate section is the ONLY pass/fail basis. Review the working tree.
1. INTEGRITY: diff swift/Sources/QwispCore/RawVerifyTests.swift vs ${LOCKDIR}/RawVerifyTests.swift — byte-identical required (restore from lock if tampered, testIntegrityOk=false).
2. STRUCTURAL: read the production encode paths — every atom ONE dispatch, actually wired (unit green ≠ wired), flag-off paths byte-unchanged, prior-wave wiring intact, fused-kernel grids match widest-parallelism member.
3. FAITHFULNESS: kernel-by-kernel math comparison vs originals (arithmetic order, precision casts, scalar half-rounding).
4. Run raw-tests (all pass), G2 per contract (byte-identical + 128/128 LOSSLESS), G5 prof per contract gates (report exact numbers; run twice if borderline).
Fix implementation defects yourself (never tests), rebuild, re-verify. Return the schema with honest numbers.`,
    { model: 'sonnet', effort: 'high', phase: 'Implement+Review', label: `review:r${round}`, schema: ROUND_SCHEMA }
  )

  log(`round ${round}: pass=${last?.pass} structural=${last?.structuralOk} g5=${(last?.g5 ?? '?').slice(0, 100)}`)
  if (last?.pass && last?.testIntegrityOk) break
  feedback = [
    `structuralOk=${last?.structuralOk} rawtests=${last?.rawtests}`,
    `g2: ${last?.g2}`, `g5: ${last?.g5}`,
    `required fixes:\n- ${(last?.requiredFixes ?? []).join('\n- ')}`,
  ].join('\n')
}

return { tests, finalReview: last, passed: !!(last?.pass && last?.testIntegrityOk) }
