import os
import json
import time
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


# -----------------------------------------------------------------------------
# 1. KAGGLE / DDP SETUP
# -----------------------------------------------------------------------------

def setup_ddp():
    # torchrun RANK/LOCAL_RANK/WORLD_SIZE verir; yoksa tek süreç (python train.py)
    if "RANK" not in os.environ:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return 0, 1, device

    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
    else:
        # Windows / CPU'da nccl yok
        device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend=backend)
    return dist.get_rank(), dist.get_world_size(), device


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def any_rank(flag, device):
    # Durma kararı bütün rank'lerde aynı olmalı, yoksa biri çıkar diğeri all_reduce'ta takılır
    if not dist.is_initialized():
        return flag
    t = torch.tensor([int(flag)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item())


# -----------------------------------------------------------------------------
# 2. TEST MODEL AND DATASET
# -----------------------------------------------------------------------------

class DummyModel(nn.Module):
    def __init__(self, dim=128, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)


def get_dummy_dataloader(per_gpu_batch, world_size, rank, seed):
    # Sabit seed: her rank aynı veriyi görür, sampler bölüştürür
    g = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.randn(10_000, 128, generator=g))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=True)
    loader = DataLoader(dataset, batch_size=per_gpu_batch, sampler=sampler, drop_last=True)
    return loader, sampler


# -----------------------------------------------------------------------------
# 3. CHECKPOINT
# -----------------------------------------------------------------------------

def save_checkpoint(out_dir, step, model, optimizer, scaler, cfg):
    # Sadece rank 0 çağırır. Önce .tmp'ye yaz, sonra rename: yarım dosya kalmaz.
    # Rotasyon: last.pt -> prev.pt, yani diskte en fazla 2 checkpoint.
    last = os.path.join(out_dir, "last.pt")
    tmp = last + ".tmp"
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "config": cfg,
    }, tmp)
    if os.path.exists(last):
        os.replace(last, os.path.join(out_dir, "prev.pt"))
    os.replace(tmp, last)


def find_resume_path(args, verbose):
    if args.resume:
        if os.path.isfile(args.resume):
            return args.resume
        if verbose:
            print(f"[uyarı] --resume bulunamadı: {args.resume} -> sıfırdan başlıyor")
    local_last = os.path.join(args.out, "last.pt")
    return local_last if os.path.isfile(local_last) else None


# -----------------------------------------------------------------------------
# 4. MAIN TRAINING
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dummy_config.yaml")
    parser.add_argument("--out", type=str, default="run")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--max-hours", type=float, default=10.5)
    parser.add_argument("--total-steps", type=int, default=None, help="config'teki total_steps'i ezer")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.total_steps is not None:
        cfg["total_steps"] = args.total_steps

    rank, world_size, device = setup_ddp()
    is_main_process = rank == 0  # sadece rank 0 yazar: checkpoint, log

    try:
        global_batch = cfg["batch_size"]
        assert global_batch % world_size == 0, f"batch_size {global_batch}, {world_size} GPU'ya bölünmüyor"
        per_gpu_batch = global_batch // world_size
        total_steps = cfg["total_steps"]
        log_every = cfg.get("log_every", 50)
        ckpt_every = cfg.get("ckpt_every", 100)
        seed = cfg.get("seed", 0)

        if is_main_process:
            os.makedirs(args.out, exist_ok=True)
            print(f"starting the training... world_size={world_size} device={device.type} "
                  f"batch={global_batch} ({per_gpu_batch}/GPU) max={args.max_hours} saat")

        torch.manual_seed(seed)
        use_amp = device.type == "cuda"  # T4: fp16 + GradScaler, CPU'da kapalı

        model = DummyModel().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3))
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        # RESUME (DDP sarmadan önce, düz modele yükle)
        start_step = 0
        resume_path = find_resume_path(args, is_main_process)
        if resume_path:
            ckpt = torch.load(resume_path, map_location="cpu")
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            if ckpt["scaler_state"]:  # CPU koşusunda scaler kapalı, state boş
                scaler.load_state_dict(ckpt["scaler_state"])
            start_step = ckpt["step"]
            if is_main_process:
                print(f"=> loaded checkpoint '{resume_path}' (step {start_step})")

        if world_size > 1:
            model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
        raw_model = model.module if isinstance(model, DDP) else model

        dataloader, sampler = get_dummy_dataloader(per_gpu_batch, world_size, rank, seed)

        # Timeout protection
        start_time = time.time()
        max_seconds = args.max_hours * 3600
        step = start_step
        last_saved_step = start_step
        epoch = step // len(dataloader)
        t_log, step_log = time.time(), step

        while step < total_steps:
            sampler.set_epoch(epoch)
            for (batch,) in dataloader:
                batch = batch.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    output = model(batch)
                loss = F.mse_loss(output.float(), batch)  # dummy loss, fp32'de hesapla

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                step += 1

                if step % log_every == 0 or step == total_steps:
                    if is_main_process:
                        it_s = (step - step_log) / (time.time() - t_log)
                        elapsed = time.time() - start_time
                        print(f"Step: {step}/{total_steps} | Loss: {loss.item():.4f} | "
                              f"{it_s:.2f} it/s | Time passed: {elapsed:.1f}s", flush=True)
                        with open(os.path.join(args.out, "log.jsonl"), "a") as f:
                            f.write(json.dumps({"step": step, "loss": loss.item(), "it_s": it_s}) + "\n")
                    t_log, step_log = time.time(), step

                if step % ckpt_every == 0 and step < total_steps:
                    if is_main_process:
                        save_checkpoint(args.out, step, raw_model, optimizer, scaler, cfg)
                        print(f"checkpoint yazıldı: step {step}", flush=True)
                    last_saved_step = step
                    if dist.is_initialized():
                        dist.barrier()

                # 12 saat sınırı: bütün rank'ler birlikte durur
                if step % log_every == 0 and any_rank(time.time() - start_time > max_seconds, device):
                    if is_main_process:
                        print("Süre doldu, güvenli çıkış yapılıyor ve checkpoint alınıyor...")
                    total_steps = step  # döngüden çık, aşağıda kaydet
                    break

                if step >= total_steps:
                    break
            epoch += 1

        # Eğitim bitti (ya da süre doldu): son checkpoint
        if step != last_saved_step:
            if is_main_process:
                save_checkpoint(args.out, step, raw_model, optimizer, scaler, cfg)
                print(f"Eğitim tamamlandı. Son checkpoint alındı: step {step} -> {args.out}/last.pt")
            if dist.is_initialized():
                dist.barrier()
        elif is_main_process and step == start_step:
            print(f"Yapılacak adım yok (step {step} >= total_steps). --total-steps ile artır.")
        elif is_main_process:
            print(f"Son checkpoint zaten güncel: step {step} -> {args.out}/last.pt")
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
