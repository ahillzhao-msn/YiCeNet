"""Train WorldModelV3 on flywheel buffer data."""
import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import json, time, torch, random, math
from pathlib import Path
from yicenet.world_model import WorldModelV3, power_law_weight_batch
from yicenet.engine_provider import EngineProvider
from yicenet.tokenizer import encode


def build_entry(s, model, device, now):
    text = s.get('user_text', '')
    ts = s.get('timestamp', now)
    tokens = s.get('token_cost', 0)
    sat = s.get('satisfaction', 0.0)
    ok = s.get('completed', False)
    corr = s.get('corrected', False)
    praised = s.get('praised', False)
    abandon = s.get('abandoned', False)
    has_hex = bool(s.get('hexagram_evolution', []))

    ids, mask = encode(text, max_len=128)
    with torch.no_grad():
        out = model(ids.to(device), mask.to(device), tau=0.01, hard=True)
        probes = out['probes'].cpu().squeeze(0)
        hex_id = out['hexagram_idx'].cpu().squeeze(0)

    has_code = 1.0 if '```' in text else 0.0
    cv = torch.tensor([
        min(len(text)/512.0, 1.0), 0.0, tokens/4096.0,
        tokens*0.3/4096.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        tokens*0.01/4000.0, has_code, 0.0,
        0.5, 0.0, 0.5, 0.3, 1.0, sat-0.5, 0.0, 0.0, 0.0,
        sat, float(corr), float(praised), float(abandon),
    ], dtype=torch.float32)
    return probes, cv, hex_id, sat, ok, has_hex, ts


def main():
    # Load data
    lines = Path.home().joinpath('.yicenet/data/flywheel_buffer.jsonl').read_text().strip().split('\n')
    all_samples = [json.loads(l) for l in lines if l.strip()]
    random.shuffle(all_samples)
    samples = [s for s in all_samples if s.get('user_text', '').strip()]
    print(f'Total: {len(all_samples)}, with text: {len(samples)}')

    # Model
    engine = EngineProvider.get_engine()
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    engine.predict('init', session_id='_train_init', turn_id=0)
    model = engine._model.to(device)
    print(f'Device: {device}, Model on: {next(model.parameters()).device}')

    now = time.time()
    print('Building entries (pre-compute probes + context vectors)...')
    t0 = time.time()
    all_entries = []
    for s in samples:
        try:
            all_entries.append(build_entry(s, model, device, now))
        except Exception:
            pass
    t1 = time.time()
    print(f'Built {len(all_entries)} entries in {t1-t0:.1f}s ({len(all_entries)/(t1-t0):.0f}/s)')

    split = int(len(all_entries) * 0.8)
    train_e, val_e = all_entries[:split], all_entries[split:]
    print(f'Train: {len(train_e)}, Val: {len(val_e)}')

    # WM
    wm = WorldModelV3(probe_dim=9, context_dim=27).to(device)
    optimizer = torch.optim.AdamW(wm.parameters(), lr=1e-4)

    EPOCHS = 15
    BATCH = 64
    best_val = float('inf')

    for ep in range(EPOCHS):
        wm.train()
        total_loss = 0.0; n_batch = 0
        random.shuffle(train_e)
        for i in range(0, len(train_e), BATCH):
            batch = train_e[i:i+BATCH]
            p_t = torch.stack([e[0] for e in batch]).to(device)
            cv_t = torch.stack([e[1] for e in batch]).to(device)
            hid_t = torch.stack([e[2] for e in batch]).to(device)
            satl = [e[3] for e in batch]
            okl = [e[4] for e in batch]
            hml = [e[5] for e in batch]
            tsl = torch.tensor([e[6] for e in batch])

            w_slow = power_law_weight_batch(tsl, now, 30.0, 1.5).to(device)
            w_fast = power_law_weight_batch(tsl, now, 3.0, 1.5).to(device)

            tgt_ext = torch.tensor([[s, 1.0 if ok else 0.0, 0.5]
                                    for s, ok in zip(satl, okl)],
                                   dtype=torch.float32, device=device)
            pred_hex, pred_ext = wm(p_t, cv_t, hid_t)

            # HeadA — mask when no hexagram data
            loss_a = torch.tensor(0.0, device=device)
            hm_t = torch.tensor(hml, dtype=torch.bool, device=device)
            if hm_t.any():
                tgt_hex = torch.zeros(len(batch), 64, device=device)
                tgt_hex[range(len(batch)), hid_t] = 1.0
                eps = 1e-8
                kl = (tgt_hex * (tgt_hex.clamp(min=eps).log() - pred_hex.clamp(min=eps).log())).sum(dim=-1)
                weighted = w_slow * kl * hm_t.float()
                loss_a = weighted.sum() / (w_slow * hm_t.float()).sum().clamp(min=eps)

            # HeadB
            se = (pred_ext - tgt_ext).pow(2).mean(dim=-1)
            loss_b = (w_fast * se).sum() / w_fast.sum().clamp(min=1e-8)
            total = loss_a + 0.3 * loss_b

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
            optimizer.step()
            total_loss += total.item()
            n_batch += 1

        # Validation
        wm.eval()
        val_mse = 0.0; val_n = 0; val_corrs = []
        with torch.no_grad():
            for e in val_e:
                p = e[0].unsqueeze(0).to(device)
                cv = e[1].unsqueeze(0).to(device)
                hid = e[2].unsqueeze(0).to(device)
                sat = e[3]
                ok = e[4]
                tgt = torch.tensor([[sat, 1.0 if ok else 0.0, 0.5]], dtype=torch.float32, device=device)
                _, pred = wm(p, cv, hid)
                val_mse += (pred - tgt).pow(2).mean().item()
                val_corrs.append((pred[0,0].item(), sat))
                val_n += 1
        val_mse /= max(val_n, 1)

        # Pearson r
        if len(val_corrs) > 2:
            preds = torch.tensor([c[0] for c in val_corrs])
            actuals = torch.tensor([c[1] for c in val_corrs])
            pm = preds.mean(); am = actuals.mean()
            num = ((preds - pm) * (actuals - am)).sum()
            den = ((preds - pm).pow(2).sum() * (actuals - am).pow(2).sum()).sqrt() + 1e-8
            r = (num / den).item()
        else:
            r = 0.0

        if val_mse < best_val:
            best_val = val_mse
            wm.save('/tmp/wm_v17_best.pt')

        avg = total_loss / max(n_batch, 1)
        print(f'Ep {ep:3d} | loss={avg:.4f} | val_mse={val_mse:.6f} | r={r:.3f} | best={best_val:.6f}')

    # Save
    ckpt_dir = Path.home() / '.yicenet' / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    wm.save(str(ckpt_dir / 'world_model_v17.pt'))
    wm.save(str(ckpt_dir / 'world_model_best.pt'))
    rmse = math.sqrt(best_val)
    print(f'\nSaved: world_model_v17.pt')
    print(f'Best val MSE: {best_val:.6f} (RMSE: {rmse:.4f})')
    print(f'Pearson r (satisfaction): {r:.3f}')
    print(f'Quality: {"PASS" if best_val < 0.05 else "WARN"}')

if __name__ == '__main__':
    main()
