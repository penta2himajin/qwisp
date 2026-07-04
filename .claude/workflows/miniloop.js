export const meta = {
  name: 'miniloop',
  description: 'Red-Green-Refactor TDD enforcing micro-loop for SMALL fixes/tasks: Opus defines goal + adds RED test(s) → Sonnet implements to GREEN (+refactor, tests stay green) → Opus audits (lock integrity + green + goal). Large implementation/verification waves use devloop instead.',
  whenToUse: '小規模な fix・対応(数十行規模)。args: {task, project:{repo,buildCmd,testCmd,testFile,doctrine?,extra?}, lockDir, maxRounds?}。大規模 wave は devloop を使うこと。',
  phases: [
    { title: 'Red', detail: 'Opus: ゴール定義 + 失敗するテスト追加(RED 証明 + write-lock)' },
    { title: 'Green', detail: 'Sonnet: 実装/作業 + refactor(テスト green 維持)' },
    { title: 'Audit', detail: 'Opus: lock 照合 + green + ゴール監査(不合格→Green へ差し戻し)' },
  ],
}

let A = args
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = null } }
if (!A || !A.task || !A.project || !A.lockDir) {
  return { aborted: 'miniloop requires args = {task, project:{repo,buildCmd,testCmd,testFile,...}, lockDir}' }
}
const P = A.project
if (!P.repo || !P.buildCmd || !P.testCmd || !P.testFile) {
  return { aborted: 'args.project requires {repo, buildCmd, testCmd, testFile}' }
}
const MAXR = A.maxRounds ?? 3

const COMMON = `
Repo: ${P.repo}. Exclusive heavy-resource discipline: you are the only build/heavy job.
Project doctrine: ${P.doctrine ?? '(none)'}
Build: ${P.buildCmd}
Test: ${P.testCmd}
${P.extra ?? ''}
Do NOT commit (the driver is the only committer).
`

const RED_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['goal', 'redProven', 'testNames', 'summary'],
  properties: {
    goal: { type: 'string', description: 'the verifiable goal statement (what GREEN + audit will check)' },
    redProven: { type: 'boolean', description: 'true iff the new test(s) FAIL on the current tree (RED observed, output quoted in summary)' },
    testNames: { type: 'string' },
    summary: { type: 'string' },
  },
}
const AUDIT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass', 'lockOk', 'testsGreen', 'goalMet', 'requiredFixes', 'summary'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if lockOk && testsGreen && goalMet' },
    lockOk: { type: 'boolean' }, testsGreen: { type: 'boolean' }, goalMet: { type: 'boolean' },
    requiredFixes: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string' },
  },
}

phase('Red')
const red = await agent(
  `${COMMON}
YOU ARE THE GOAL/TEST AUTHOR (Red phase of Red-Green-Refactor TDD). TASK: ${A.task}
1. Define the smallest VERIFIABLE goal for this task (state it precisely).
2. Add failing test(s) that encode the goal into ${P.testFile} following the project's existing test idiom (references = production code paths, no reimplemented oracles; stub-RED pattern if new APIs are needed — stubs return nil/fail, NEVER delegate to existing working code).
3. Build + run tests: PROVE the new test(s) FAIL on the current tree (quote the failure line) and pre-existing tests still pass.
4. WRITE-LOCK: mkdir -p ${A.lockDir} && cp ${P.testFile} ${A.lockDir}/
Return the schema. Your tests are IMMUTABLE for the rest of the run.`,
  { model: 'opus', effort: 'high', phase: 'Red', label: 'red', schema: RED_SCHEMA }
)
log(`red: proven=${red?.redProven} tests=${red?.testNames}`)
if (!red?.redProven) return { aborted: 'RED not proven — refusing to proceed (GREEN-by-delegation guard)', red }

let feedback = ''
let audit = null
for (let round = 1; round <= MAXR; round++) {
  phase('Green')
  const green = await agent(
    `${COMMON}
YOU ARE THE IMPLEMENTER (Green phase), round ${round}. TASK: ${A.task}
GOAL (fixed by Red phase): ${red.goal}
Tests to satisfy (IMMUTABLE — do NOT modify ${P.testFile}): ${red.testNames}
${round === 1 ? '' : `AUDIT FEEDBACK to fix:\n${feedback}\n`}
Implement the smallest change that makes the tests pass, then REFACTOR for clarity/idiom while keeping all tests green (run build+tests after each stage). If a locked test contradicts production reality, STOP and report — never bend semantics to pass.
Report: what changed (files/functions), final test output line, refactor notes.`,
    { model: 'sonnet', effort: 'medium', phase: 'Green', label: `green:r${round}` }
  )

  phase('Audit')
  audit = await agent(
    `${COMMON}
YOU ARE THE AUDITOR (Opus), round ${round}. TASK: ${A.task} / GOAL: ${red.goal}
1. LOCK: diff ${P.testFile} vs ${A.lockDir}/$(basename ${P.testFile}) — byte-identical required (restore from lock + set lockOk=false if tampered).
2. GREEN: build + run tests yourself — all pass, including the Red tests (${red.testNames}).
3. GOAL: verify the goal is genuinely met (read the diff critically — no semantic bending, no delegation shortcuts, production path actually changed as intended).
Implementer report: ${String(green).slice(0, 1500)}
Return the schema (requiredFixes concrete if failing).`,
    { model: 'opus', effort: 'high', phase: 'Audit', label: `audit:r${round}`, schema: AUDIT_SCHEMA }
  )
  log(`round ${round}: pass=${audit?.pass} green=${audit?.testsGreen} goal=${audit?.goalMet}`)
  if (audit?.pass) break
  feedback = `- ${(audit?.requiredFixes ?? []).join('\n- ')}`
}

return { red, finalAudit: audit, passed: !!audit?.pass }
