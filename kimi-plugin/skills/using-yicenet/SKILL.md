# Using YiCeNet (易策网络)

YiCeNet is a 5.6M-parameter, fully-local neural network inspired by the I-Ching (易经). It maps user intent to one of 64 hexagrams and produces an explainable orchestration decision in ~4ms.

This plugin injects YiCeNet context automatically on every turn. You will see a JSON object under the key `yicenet` in the context when the hook fires successfully.

## What YiCeNet tells you

Each prediction contains:

| Field | Meaning | How to use it |
|---|---|---|
| `hexagram` | Selected hexagram name, e.g. 乾、坤、屯、解 | Treat as a structural hint for the turn's decision archetype |
| `action` | Recommended action primitive, e.g. `parallel_invoke`, `auth_check`, `decompose` | Consider it alongside your own judgment; do not follow blindly |
| `env_confidence` | 0.0–1.0 routing confidence | ≥0.7: strong signal; 0.4–0.7: moderate; <0.4: weak — rely on your normal reasoning |
| `context_status` | `sufficient` / `partial` / `thin` | `partial`/`thin` means you lack context — ask clarifying questions or summarize old turns |
| `prescription.retain_turns` | Turn indices to keep fully | Keep these in working memory |
| `prescription.summarize_turns` | Turn indices to compress | Summarize to 1–3 bullet points |
| `prescription.discard_turns` | Turn indices to drop | Do not let them consume context |

## Hexagram quick reference

- **乾 (Qián)** — strong, initiating action; good for starting or leading
- **坤 (Kūn)** — receptive, supporting; listen, gather context, defer
- **屯 (Zhūn)** — initial difficulty; break the task into smaller steps
- **蒙 (Méng)** — ignorance/learning; explain, teach, ask questions
- **需 (Xū)** — waiting; pause, verify prerequisites
- **讼 (Sòng)** — conflict; reconcile, clarify disagreement
- **师 (Shī)** — organize; marshal resources, plan systematically
- **比 (Bǐ)** — alliance; reuse existing work, collaborate
- **小畜 (Xiǎo Chù)** — small accumulation; incremental progress
- **履 (Lǚ)** — careful conduct; verify assumptions before acting
- **泰 (Tài)** — harmony; integrate, balance perspectives
- **否 (Pǐ)** — stagnation; diagnose blockers first
- **解 (Jiě)** — release of tension; resolve, simplify, clear debt
- **恒 (Héng)** — perseverance; stay the course

## Rules of engagement

1. **YiCeNet is a hint, not an order.** The hexagram and action are structural suggestions. Override them when engineering judgment demands it.
2. **Use confidence as a weight.** High `env_confidence` → let the hint shape phrasing and priorities. Low confidence → ignore it and proceed normally.
3. **Respect context prescriptions.** If YiCeNet says context is `thin`, ask for missing information or summarize prior turns before acting.
4. **Do not explain YiCeNet mechanics to the user unless asked.** The user sees only the result, not the routing pipeline.
5. **Prefer the explicit MCP tools for deep decisions.** When facing a major architectural or prioritization choice, call `mcp__yicenet__yicenet_predict` explicitly to get the full candidate list and Q-values.

## Explicit MCP tools available

- `mcp__yicenet__yicenet_predict` — full hexagram routing + context prescription
- `mcp__yicenet__yicenet_attend` — lightweight context pruning recommendation
- `mcp__yicenet__yicenet_turn_complete` — record response metadata (called automatically by hooks)
- `mcp__yicenet__yicenet_feedback` — submit explicit reward signal
- `mcp__yicenet__yicenet_switch` — hot-swap to another checkpoint

## When to call `mcp__yicenet__yicenet_predict` explicitly

- The user asks for a strategic decision between multiple approaches.
- You are about to start a complex multi-step task and want a structural framing.
- The injected hook context was `thin` and you want a richer read after gathering more information.

## Fallback if hooks are disabled

If the automatic hooks are not firing, you can still use YiCeNet by calling `mcp__yicenet__yicenet_predict` at the start of a turn and `mcp__yicenet__yicenet_turn_complete` after your response.
