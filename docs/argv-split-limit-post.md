<!--
  DRAFT — NOT VERIFIED. Before publishing:
  1. Every command-output block below is RECONSTRUCTED, not from a real run.
     Run each on the real code and replace: invocation, score, latency,
     rule name, policy line.
  2. Confirm the exact subcommand for checking a tool CALL (c2s direction).
     It may differ from the `detect` used for tool RESULTS (s2c) in the last post.
  3. Confirm the failing-test path and class name (placeholder: TestArgvSplitGap).
  4. Confirm the current test count if you cite it anywhere.
  Remove this comment before publishing.
-->

# I tried to patch a blind spot in my own MCP tool. The patch and a false-positive bug cancel out.

I maintain [bulwark-mcp](https://github.com/churik5/bulwark-mcp), a local proxy that sits between an MCP client (Claude Desktop, Cursor) and the servers it talks to, and screens the traffic for dangerous patterns. A few weeks ago I wrote about [a class of prompt injection it can't catch](https://dev.to/churik5/i-tried-to-break-my-own-mcp-prompt-injection-detector-one-class-of-attack-walks-straight-through--4534). This is a companion finding from a different layer — the part that inspects tool *calls*, not tool *results* — and it lands in the same uncomfortable place: the thing you'd reach for to fix it turns out to be unfixable, and the reason is worth more than the bug.

## What the rule does

One job an MCP proxy can take on is refusing obviously dangerous tool calls before they reach a server. If an agent — hijacked, or just confused — tries to call a shell tool with `rm -rf /`, you'd like that stopped. bulwark has shell-command rules for exactly this, and on the direct form they work:

```
$ bulwark detect '["rm", "-rf", "/"]'
BLOCK (score=0.90, latency=0 ms)
rules hit:
  • shell.destructive.rm_rf
policy: block_high_score_c2s → block
```

A flat argv array with `rm` next to `-rf` next to `/` is a shape the pattern knows. Correctly blocked.

Then I tried the same command a different way.

## The gap

MCP tool calls are JSON, and "run `rm` with these arguments" has no single canonical shape. It can be a flat array:

```json
["rm", "-rf", "/"]
```

or the exact same command can be split across named fields:

```json
{"cmd": "rm", "args": ["-rf", "/"]}
```

Same binary, same flags, same destruction. But the rule matches a *contiguous token sequence* — it's looking for `rm` followed by `-rf` followed by `/`. In the split form there is no such sequence: `rm` lives under `cmd`, the flags live under `args`, and nothing in the raw text ever puts them next to each other.

```
$ bulwark detect '{"cmd": "rm", "args": ["-rf", "/"]}'
PASS (score=0.00, latency=0 ms)
rules: no hit
policy: no match → allow
```

Score 0.00. It walks straight through. So I sat down to fix it — and that's where it fell apart.

## Why the fix is impossible

To catch the split form, the rule has to stop matching flat token sequences and start reconstructing *"this is a command named `rm` being handed dangerous flags"* out of an arbitrary field structure.

The problem: legitimate tools use arbitrary field structures too. A build tool takes `{"command": "build", "target": "//app:server"}`. A database tool takes `{"op": "delete", "table": "sessions", "where": {...}}`. There is no structural feature that separates "malicious command in named fields" from "legitimate command in named fields," because it *is* the same shape. Any rule broad enough to catch the split `rm` also flags the legitimate multi-field calls.

So the two requirements pull in opposite directions. *Catch the split form* and *don't flag legitimate calls* aren't two bugs I can fix one after the other — they're the same dial. Tighten the pattern to catch the encoding and false positives climb; loosen it to spare real calls and the encoding walks through. I can trade one for the other. I can't have both. This isn't a regex I haven't written yet — it's a regex that provably can't exist.

## The lesson underneath it

Argument inspection can't be the trust boundary for *"is this tool call dangerous,"* for the same reason content inspection can't be the boundary for *"is this text an instruction"* — the subject of the last post. The danger isn't in the surface form of the arguments; it's in what the tool *does* when it runs. And a single action has unbounded encodings: flat, nested, split, renamed fields, base64'd values — whatever the schema allows. You can't enumerate encodings any more than you can enumerate the phrasings of "ignore all previous instructions." Both are open sets, and the attacker gets to pick from them second.

## What actually holds

The control that doesn't have this problem ignores the arguments entirely. bulwark's capability allowlist matches on the tool *name*: if a workflow is pinned to `filesystem.read` and `github.create_issue`, then a call to `shell.exec` is refused — not because its arguments looked bad, but because `shell.exec` isn't on the list at all. Every encoding of `rm -rf /` dies at the same gate, because the gate never reads the arguments. The malicious-encoding problem disappears because the question changed: from *"are these arguments dangerous?"* (unbounded, unanswerable) to *"is this tool allowed here?"* (finite, and mine to define).

I want to be precise about what that does and doesn't buy you. The allowlist is coarse — exact name match, no argument awareness. It's off until you configure it. And it only helps when the dangerous call is to a tool that shouldn't be reachable in the first place; if the tool is already allowed and an injection abuses it, the allowlist does nothing. It shrinks what a hijack can reach; it does not inspect what it does. Like everything here, it's one layer, defeatable on its own.

## It's a failing test, not a footnote

I didn't fix the shell rule, because I now think fixing it is the wrong thing to want. Instead the gap is pinned as an executable spec — `TestArgvSplitGap` in `tests/test_detectors_rules.py` — asserting the detector currently misses the split form. The day someone finds a rule that catches the encoding *without* lighting up on legitimate multi-field calls, that test goes red — and red would mean the impossible turned out to be possible. I would be glad to be wrong.

bulwark-mcp is AGPL-3.0, Python, runs entirely locally, and sends nothing anywhere by default. It's firmly v0.x, and the detector ships off by default — on purpose. If you've got a shell tool call in a shape the rules miss, or — better — an actual argument-level control that doesn't trade catch-rate for false positives, opening an issue is the single most useful thing you could do. Repo and the test above: https://github.com/churik5/bulwark-mcp
