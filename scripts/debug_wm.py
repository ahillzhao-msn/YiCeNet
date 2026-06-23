"""Debug WMv3 training - single batch check."""
import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import json, torch, time
from pathlib import Path
from yicenet.world_model import WorldModelV3, power_law_weight_batch
from yicenet.engine_provider import EngineProvider
from yicenet.tokenizer import encode

# Load engine + model
engine = EngineProvider.get_engine()
result = engine.predict('hello', session_id='dbg', turn_id=0)
model = engine._model
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)
model = model.to(device)
print(f'Device: {device}')

# Load 2 samples
lines = Path.home().joinpath('.yicenet/data/flywheel_buffer.jsonl').read_text().strip().split('\n')
batch = [json.loads(lines[i]) for i in range(2)]
print(f'Sample 0 sat={batch[0]["satisfaction"]} ok={batch[0]["completed"]}')
print(f'Sample 1 sat={batch[1]["satisfaction"]} ok={batch[1]["completed"]}')

now = time.time()
pl, cvl, hidl, satl, okl, tsl = [], [], [], [], [], []
for s in batch:
    text = s.get('user_text', '')
    sat = s.get('satisfaction', 0.0)
    ok = s.get('completed', False)
    ids, mask = encode(text, max_len=128)
    ids, mask = ids.to(device), mask.to(device)
    with torch.no_grad():
        out = model(ids, mask, tau=0.01, hard=True)
        probes = out['probes'].cpu().squeeze(0)
        hex_id = out['hexagram_idx'].cpu().squeeze(0)
    
    tokens = s.get('token_cost', 0)
    has_code = 1.0 if '```' in text else 0.0
    corr = s.get('corrected', False)
    praised = s.get('praised', False)
    abandon = s.get('abandoned', False)
    ts = s.get('timestamp', now)
    
    cv = torch.tensor([
        min(len(text)/512.0, 1.0), 0.0, tokens/4096.0,
        tokens*0.3/4096.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        tokens*0.01/4000.0, has_code, 0.0,
        0.5, 0.0, 0.5, 0.3, 1.0, sat-0.5, 0.0, 0.0, 0.0,
        sat, float(corr), float(praised), float(abandon),
    ], dtype=torch.float32)
    
    pl.append(probes); cvl.append(cv); hidl.append(hex_id)
    satl.append(sat); okl.append(ok); tsl.append(ts)

print(f'\n=== Batch data ===')
print(f'probes[0]: {pl[0][:5]}...')
print(f'cv[0][:5]: {cvl[0][:5]}...')
print(f'hex_id: {hidl}')

# WM forward
wm = WorldModelV3(probe_dim=9, context_dim=27).to(device)
p_t = torch.stack(pl).to(device)
cv_t = torch.stack(cvl).to(device)
hid_t = torch.stack(hidl).to(device)
print(f'\nInput shapes: p={p_t.shape}, cv={cv_t.shape}, hid={hid_t.shape}')

pred_hex, pred_ext = wm(p_t, cv_t, hid_t)
print(f'pred_ext: {pred_ext}')
print(f'pred_hex sums: {pred_hex.sum(dim=-1)}')

# HeadB target
tgt_ext = torch.tensor([[sat, 1.0 if ok else 0.0, 0.5] for sat, ok in zip(satl, okl)], dtype=torch.float32).to(device)
print(f'tgt_ext: {tgt_ext}')

se = (pred_ext - tgt_ext).pow(2).mean(dim=-1)
print(f'se: {se}')

w_fast = power_law_weight_batch(torch.tensor(tsl), now, 3.0, 1.5).to(device)
print(f'w_fast: {w_fast}')
print(f'w_fast sum: {w_fast.sum()}')

loss_b = (w_fast * se).sum() / w_fast.sum().clamp(min=1e-8)
print(f'loss_b: {loss_b}')

# HeadA
loss_a = torch.tensor(0.0, device=device)
print(f'loss_a: {loss_a}')

total = loss_a + 0.3 * loss_b
print(f'total loss: {total}')
print(f'total.item(): {total.item()}')

optimizer = torch.optim.AdamW(wm.parameters(), lr=1e-4)
optimizer.zero_grad()
total.backward()
print(f'Grad norm: {sum(p.grad.norm().item() for p in wm.parameters() if p.grad is not None):.4f}')
