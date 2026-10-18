from xml.parsers.expat import model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy
from tqdm import trange
import time
import numpy as np
import json
import os
from pathlib import Path
import statistics
import copy

import torch
import torch.nn as nn

# --- MLP-Mixer modules ---
class MLP(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_features, bias=False)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class MixerBlock(nn.Module):
    def __init__(
        self,
        num_patches, dim,
        token_dim, token_out_dim,
        channel_dim, channel_out_dim,
    ):
        super().__init__()
        # --- Token mixing ---
        self.norm1 = nn.LayerNorm(dim)
        self.token_mlp = MLP(num_patches, token_dim, token_out_dim)
        # --- Channel mixing ---
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mlp = MLP(dim, channel_dim, channel_out_dim)

    def forward(self, x):
        # x: [B, N, D]
        y = self.norm1(x).transpose(1, 2)          # [B, D, N]
        token_out = self.token_mlp(y)              # [B, N, token_out_dim]
        token_out = token_out.transpose(1, 2)      # [B, N, D]
        x = x + token_out                          # residual

        z = self.norm2(x)
        channel_out = self.channel_mlp(z)          # [B, N, channel_out_dim]
        x = x + channel_out                        # residual
        return x

class MLPMixer(nn.Module):
    def __init__(
        self,
        patch_size=4,
        in_channels=3,
        embedding_dim=128,
        num_classes=10,
        depth=8,
        token_dim=64,
        channel_dim=512,
        mixer_block_dims=None,  # list of tuples: see below!
    ):
        super().__init__()
        patch_dim = patch_size * patch_size * in_channels
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embedding = nn.Linear(patch_dim, embedding_dim)
        # Alapértelmezett architektúra (nincs pruning):
        if mixer_block_dims is None:
            num_patches = (32 // patch_size) ** 2  # pl. MNIST: 32x32
            mixer_block_dims = [
                # (num_patches, embedding_dim, token_dim, token_out_dim,
                #  channel_dim, channel_out_dim)
                (num_patches, embedding_dim, token_dim, num_patches,
                 channel_dim, embedding_dim)
                for _ in range(depth)
            ]

        self.mixer_blocks = nn.Sequential(*[
            MixerBlock(
                num_patches=cfg[0],
                dim=cfg[1],
                token_dim=cfg[2],
                token_out_dim=cfg[3],
                channel_dim=cfg[4],
                channel_out_dim=cfg[5],
            ) for cfg in mixer_block_dims
        ])
        self.norm = nn.LayerNorm(mixer_block_dims[-1][1])
        self.classifier = nn.Linear(mixer_block_dims[-1][1], num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)             # [B, C, nH, nW, p, p]
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()      # [B, nH, nW, C, p, p]
        x = x.view(B, -1, C * p * p)                      # [B, num_patches, patch_dim]
        x = self.embedding(x)                             # [B, num_patches, embedding_dim]
        x = self.mixer_blocks(x)                          # [B, num_patches, embedding_dim]
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.classifier(x)

# --- LTH components ---
def apply_masks(model, masks):
    """Keep removed neurons zero during optimizer updates."""
    with torch.no_grad():
        for block, mask in zip(model.mixer_blocks, masks):
            for name in ('token_mlp', 'channel_mlp'):
                mlp = getattr(block, name)
                mlp.fc1.weight *= mask[name + '.fc1']
                mlp.fc2.weight *= mask[name + '.fc2']
                if mlp.fc1.bias is not None:
                    mlp.fc1.bias *= (mask[name + '.fc1'].sum(dim=1) > 0)


def train(model, optimizer, dataloader, device, masks=None):
    model.train()
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if masks is not None:
            apply_masks(model, masks)

def test(model, dataloader, device):
    model.eval()
    correct = 0
    total_infer_time = 0.0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            output = model(data)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            total_infer_time += time.perf_counter() - start
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return correct / len(dataloader.dataset), total_infer_time


def correlation_mask(weight_tensor: torch.Tensor, global_mask=None, relative_margin=0.15, verbose=False, layer_name="layer", axis=0):
    """
    Általános korrelációalapú maszkolás.
    - axis=0: sorok (kimeneti neuronok) prunolása
    - axis=1: oszlopok (bemeneti neuronok) prunolása
    """
    W = weight_tensor.detach().cpu().numpy()

    if global_mask is not None:
        W = W * global_mask.cpu().numpy()

    if axis == 0:
        norms = np.linalg.norm(W, axis=1)
        nonzero_indices = np.where(norms > 1e-6)[0]
        data_for_corr = W[nonzero_indices]
    elif axis == 1:
        norms = np.linalg.norm(W, axis=0)
        nonzero_indices = np.where(norms > 1e-6)[0]
        data_for_corr = W[:, nonzero_indices].T
    else:
        raise ValueError("axis must be 0 (rows) or 1 (columns)")

    if len(nonzero_indices) < 2:
        if verbose:
            print(f"  [Debug] {layer_name}: kevés nem-nulla {'sor' if axis==0 else 'oszlop'} maradt, nincs értelme korrelációt számolni.")
        return global_mask.clone() if global_mask is not None else torch.ones_like(weight_tensor)

    corr_matrix = np.corrcoef(data_for_corr)
    np.fill_diagonal(corr_matrix, 0.0)

    max_corr = np.nanmax(np.abs(corr_matrix))
    threshold = max(max_corr - relative_margin, 0.0)

    if verbose:
        print(f"  [Debug] {layer_name}: max(abs(corr)) = {max_corr:.4f}, threshold = {threshold:.4f}")

    mask = np.ones_like(W)
    already_pruned = set()

    for i in range(len(nonzero_indices)):
        for j in range(i + 1, len(nonzero_indices)):
            if abs(corr_matrix[i, j]) > threshold:
                j_idx = int(nonzero_indices[j])
                if j_idx not in already_pruned:
                    if axis == 0:
                        mask[j_idx, :] = 0.0
                    else:
                        mask[:, j_idx] = 0.0
                    already_pruned.add(j_idx)

    if verbose:
        print(f"  [Debug] {layer_name}: nullázva ({'sor' if axis==0 else 'oszlop'} indexek): {sorted(list(already_pruned))}")

    final_mask = torch.tensor(mask, dtype=torch.float32, device=weight_tensor.device)
    if global_mask is not None:
        final_mask *= global_mask
    return final_mask

def propagate_mask(prev_mask: torch.Tensor, next_weight: torch.Tensor) -> torch.Tensor:
    row_sums = prev_mask.sum(dim=1)
    zero_rows = (row_sums == 0).nonzero(as_tuple=False).squeeze()
    next_mask = torch.ones_like(next_weight)
    if zero_rows.numel() > 0:
        next_mask[:, zero_rows] = 0.0
    return next_mask

def prune_mixer_model(model, global_masks, relative_margin=0.15, verbose=False):
    """
    Korrelációalapú pruning minden MixerBlock token_mlp és channel_mlp részén.
    FC1: sorok nullázása, FC2: oszlopok nullázása.
    """
    new_global_masks = []

    for i, block in enumerate(model.mixer_blocks):
        device = block.token_mlp.fc1.weight.device

        # === TOKEN MLP FC1 ===
        mask_token_mlp_fc1 = correlation_mask(
            block.token_mlp.fc1.weight,
            global_mask=global_masks[i]['token_mlp.fc1'],
            relative_margin=relative_margin,
            verbose=verbose,
            layer_name=f"token_mlp_fc1_block_{i}",
            axis=0
        )

        # === TOKEN MLP FC2 input: fc1 output alapján prune ===
        mask_token_mlp_fc2 = global_masks[i]['token_mlp.fc2'] * propagate_mask(mask_token_mlp_fc1, block.token_mlp.fc2.weight).to(device)

        # === TOKEN MLP FC2 input (axis=1) korrelációs pruning ===
        mask_token_mlp_fc2 = correlation_mask(
            block.token_mlp.fc2.weight,
            global_mask=mask_token_mlp_fc2,
            relative_margin=relative_margin,
            verbose=verbose,
            layer_name=f"token_mlp_fc2_block_{i}",
            axis=1
        )

        # === CHANNEL MLP ===
        mask_channel_mlp_fc1 = correlation_mask(
            block.channel_mlp.fc1.weight,
            global_mask=global_masks[i]['channel_mlp.fc1'],
            relative_margin=relative_margin,
            verbose=verbose,
            layer_name=f"channel_mlp_fc1_block_{i}",
            axis=0
        )
        mask_channel_mlp_fc2 = global_masks[i]['channel_mlp.fc2'] * propagate_mask(mask_channel_mlp_fc1, block.channel_mlp.fc2.weight).to(device)
        
        mask_channel_mlp_fc2 = correlation_mask(
            block.channel_mlp.fc2.weight,
            global_mask=mask_channel_mlp_fc2,
            relative_margin=relative_margin,
            verbose=verbose,
            layer_name=f"channel_mlp_fc2_block_{i}",
            axis=1
        )
        
        bias_mask_token_mlp_fc1 = (mask_token_mlp_fc1.sum(dim=1) != 0).float()
        bias_mask_channel_mlp_fc1 = (mask_channel_mlp_fc1.sum(dim=1) != 0).float()

        # === APPLY ALL MASKS ===
        with torch.no_grad():
            block.token_mlp.fc1.weight *= mask_token_mlp_fc1
            if block.token_mlp.fc1.bias is not None:
                block.token_mlp.fc1.bias *= bias_mask_token_mlp_fc1
            
            block.token_mlp.fc2.weight *= mask_token_mlp_fc2
            
            block.channel_mlp.fc1.weight *= mask_channel_mlp_fc1
            if block.channel_mlp.fc1.bias is not None:
                block.channel_mlp.fc1.bias *= bias_mask_channel_mlp_fc1
            
            block.channel_mlp.fc2.weight *= mask_channel_mlp_fc2

        # === UPDATE GLOBAL MASKS ===
        new_global_masks.append({
            'token_mlp.fc1': global_masks[i]['token_mlp.fc1'] * mask_token_mlp_fc1,
            'token_mlp.fc1.bias': (
                global_masks[i]['token_mlp.fc1.bias'] * bias_mask_token_mlp_fc1
                if global_masks[i]['token_mlp.fc1.bias'] is not None
                else None
            ),
            
            'token_mlp.fc2': global_masks[i]['token_mlp.fc2'] * mask_token_mlp_fc2,
            
            'channel_mlp.fc1': global_masks[i]['channel_mlp.fc1'] * mask_channel_mlp_fc1,
            'channel_mlp.fc1.bias': (
                global_masks[i]['channel_mlp.fc1.bias'] * bias_mask_channel_mlp_fc1
                if global_masks[i]['channel_mlp.fc1.bias'] is not None
                else None
            ),
            
            'channel_mlp.fc2': global_masks[i]['channel_mlp.fc2'] * mask_channel_mlp_fc2
        })

    return new_global_masks


def build_fully_pruned_mixer_model(model, global_masks):
    device = next(model.parameters()).device
    mixer_block_dims = []

    # Kezdeti bemeneti dimenziók (pl. CIFAR10: 32x32 patch -> 16)
    orig_num_patches = (32 // model.patch_size) ** 2
    orig_embedding_dim = model.embedding.out_features

    current_token_dim = orig_num_patches
    current_channel_dim = orig_embedding_dim

    for i, block in enumerate(model.mixer_blocks):
        # --- Token MLP pruning ---
        mask_token_fc1 = global_masks[i]['token_mlp.fc1']
        mask_token_fc2 = global_masks[i]['token_mlp.fc2']
        token_fc1_in = (mask_token_fc1.sum(dim=0) > 0).nonzero(as_tuple=True)[0]
        token_fc1_out = (mask_token_fc1.sum(dim=1) > 0).nonzero(as_tuple=True)[0]
        token_fc2_in = (mask_token_fc2.sum(dim=0) > 0).nonzero(as_tuple=True)[0]
        token_hidden = token_fc1_out[torch.isin(token_fc1_out, token_fc2_in)]

        # --- Channel MLP pruning ---
        mask_channel_fc1 = global_masks[i]['channel_mlp.fc1']
        mask_channel_fc2 = global_masks[i]['channel_mlp.fc2']
        channel_fc1_in = (mask_channel_fc1.sum(dim=0) > 0).nonzero(as_tuple=True)[0]
        channel_fc1_out = (mask_channel_fc1.sum(dim=1) > 0).nonzero(as_tuple=True)[0]
        channel_fc2_in = (mask_channel_fc2.sum(dim=0) > 0).nonzero(as_tuple=True)[0]
        channel_hidden = channel_fc1_out[torch.isin(channel_fc1_out, channel_fc2_in)]

        # Pruned MLP méretek (csak a rejtett dim csökken)
        token_dim = len(token_hidden)
        token_out_dim = current_token_dim  # residual miatt fix
        channel_dim = len(channel_hidden)
        channel_out_dim = current_channel_dim  # residual miatt fix

        # Block konfigurációk
        mixer_block_dims.append((
            current_token_dim, current_channel_dim,
            token_dim, token_out_dim,
            channel_dim, channel_out_dim
        ))

    pruned_mixer = MLPMixer(
        patch_size=model.patch_size,
        in_channels=model.in_channels,
        embedding_dim=model.embedding.out_features,
        num_classes=model.classifier.out_features,
        mixer_block_dims=mixer_block_dims
    ).to(device)

    pruned_mixer.embedding.load_state_dict(model.embedding.state_dict())
    pruned_mixer.norm.load_state_dict(model.norm.state_dict())
    pruned_mixer.classifier.load_state_dict(model.classifier.state_dict())

    # Súlyok másolása prunolt FC-kre!
    for i, block in enumerate(pruned_mixer.mixer_blocks):
        orig_block = model.mixer_blocks[i]
        with torch.no_grad():
            # --- Token MLP ---
            mask_token_fc1 = global_masks[i]['token_mlp.fc1']
            mask_token_fc2 = global_masks[i]['token_mlp.fc2']
            token_fc1_in = (mask_token_fc1.sum(dim=0) > 0).nonzero(as_tuple=True)[0]
            token_fc1_out = (mask_token_fc1.sum(dim=1) > 0).nonzero(as_tuple=True)[0]
            token_fc2_in = (mask_token_fc2.sum(dim=0) > 0).nonzero(as_tuple=True)[0]
            token_hidden = token_fc1_out[torch.isin(token_fc1_out, token_fc2_in)]

            # FC1 (pruned in-dim ellenőrzés)
            orig_in_dim = orig_block.token_mlp.fc1.in_features
            pruned_in_dim = block.token_mlp.fc1.in_features
            if pruned_in_dim == len(token_fc1_in):
                block.token_mlp.fc1.weight.copy_(
                    orig_block.token_mlp.fc1.weight[token_hidden][:, token_fc1_in]
                )
            else:
                block.token_mlp.fc1.weight.copy_(
                    orig_block.token_mlp.fc1.weight[token_hidden]
                )
            if block.token_mlp.fc1.bias is not None:
                block.token_mlp.fc1.bias.copy_(orig_block.token_mlp.fc1.bias[token_hidden])

            # FC2 (out fix, in: pruned hidden)
            block.token_mlp.fc2.weight.copy_(
                orig_block.token_mlp.fc2.weight[:, token_hidden]
            )
            if block.token_mlp.fc2.bias is not None:
                block.token_mlp.fc2.bias.copy_(orig_block.token_mlp.fc2.bias)

            # --- Channel MLP ---
            mask_channel_fc1 = global_masks[i]['channel_mlp.fc1']
            mask_channel_fc2 = global_masks[i]['channel_mlp.fc2']
            channel_fc1_in = (mask_channel_fc1.sum(dim=0) > 0).nonzero(as_tuple=True)[0]
            channel_fc1_out = (mask_channel_fc1.sum(dim=1) > 0).nonzero(as_tuple=True)[0]
            channel_fc2_in = (mask_channel_fc2.sum(dim=0) > 0).nonzero(as_tuple=True)[0]
            channel_hidden = channel_fc1_out[torch.isin(channel_fc1_out, channel_fc2_in)]

            # FC1 (pruned in-dim ellenőrzés)
            orig_in_dim = orig_block.channel_mlp.fc1.in_features
            pruned_in_dim = block.channel_mlp.fc1.in_features
            if pruned_in_dim == len(channel_fc1_in):
                block.channel_mlp.fc1.weight.copy_(
                    orig_block.channel_mlp.fc1.weight[channel_hidden][:, channel_fc1_in]
                )
            else:
                block.channel_mlp.fc1.weight.copy_(
                    orig_block.channel_mlp.fc1.weight[channel_hidden]
                )
            if block.channel_mlp.fc1.bias is not None:
                block.channel_mlp.fc1.bias.copy_(orig_block.channel_mlp.fc1.bias[channel_hidden])

            # FC2 (out fix, in: pruned hidden)
            block.channel_mlp.fc2.weight.copy_(
                orig_block.channel_mlp.fc2.weight[:, channel_hidden]
            )
            if block.channel_mlp.fc2.bias is not None:
                block.channel_mlp.fc2.bias.copy_(orig_block.channel_mlp.fc2.bias)

            # Norm rétegek
            block.norm1.load_state_dict(orig_block.norm1.state_dict())
            block.norm2.load_state_dict(orig_block.norm2.state_dict())
            # Projekciós layereket nem kell másolni, random marad!

    print("Num of parameters in pruned model:", sum(p.numel() for p in pruned_mixer.parameters()))
    print("Num of parameters in original model:", sum(p.numel() for p in model.parameters()))
    return pruned_mixer



DEFAULT_CIFAR10_URL = (
    'https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python/'
    'resolve/a48007227f9e2cd0af96175f4afb5ac4965e261b/cifar-10-python.tar.gz'
)


class MirrorCIFAR10(datasets.CIFAR10):
    """Use the mirror with torchvision's original archive/file MD5 checks."""
    def __init__(self, *args, **kwargs):
        self.url = os.environ.get('MIXER_CIFAR10_URL') or DEFAULT_CIFAR10_URL
        super().__init__(*args, **kwargs)


def compact_and_verify(model, masks, example):
    """Physically remove masked neurons, verifying the transformation."""
    model.eval()
    apply_masks(model, masks)
    compact = build_fully_pruned_mixer_model(model, masks).eval()
    with torch.inference_mode():
        expected = model(example)
        actual = compact(example)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    return compact


def benchmark_models(original, pruned, batch_sizes=(1, 128, 1024),
                     warmup=20, iterations=50, rounds=7, output_path='benchmark_results.json',
                     include_compiled=None, masked=None):
    """Compare eager and reduce-overhead inference with compilation outside timing.

    Wall latency includes Python/kernel launch overhead; CUDA-event latency is
    also recorded. Dataset loading and host-to-device transfers are excluded.
    """
    device = next(original.parameters()).device
    if next(pruned.parameters()).device != device:
        raise ValueError('Both models must be on the same device')
    if warmup < 0 or iterations < 1 or rounds < 1 or not batch_sizes or any(b < 1 for b in batch_sizes):
        raise ValueError('Invalid benchmark configuration')
    models = {'original': original.eval().float(), 'pruned': pruned.eval().float()}
    if masked is not None:
        if next(masked.parameters()).device != device:
            raise ValueError('Masked model must be on the same device')
        models['masked'] = masked.eval().float()
    compile_requested = device.type == 'cuda' if include_compiled is None else include_compiled
    counts = {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()}
    reduction = 1 - counts['pruned'] / counts['original']
    result = {
        'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'dtype': 'float32', 'execution': 'eager', 'includes_data_loading': False,
        'tf32_matmul_allowed': torch.backends.cuda.matmul.allow_tf32,
        'warmup': warmup, 'iterations_per_round': iterations, 'rounds': rounds,
        'parameters': counts, 'parameter_reduction_fraction': reduction,
        'original_depth': len(original.mixer_blocks), 'pruned_depth': len(pruned.mixer_blocks),
        'at_least_50_percent_smaller': reduction >= 0.5,
        'pruned_block_hidden_dims': [
            [b.token_mlp.fc1.out_features, b.channel_mlp.fc1.out_features]
            for b in pruned.mixer_blocks
        ],
        'batches': [],
        'masked_control': {'present': masked is not None, 'outputs_verified': False,
                           'representation': 'Original dense dimensions, removed weights zeroed; no runtime mask multiplication.'},
        'compiled': {'mode': 'reduce-overhead', 'fullgraph': True, 'dynamic': False,
                     'compilation_in_inference_time': False, 'batches': [], 'errors': [],
                     'status': 'pending' if compile_requested else 'skipped',
                     'skip_reason': None if compile_requested else 'Compiled GPU comparison not requested on this device.',
                     'cuda_graph_capture_verified': False},
    }

    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def measure(model, x):
        start_event = end_event = None
        if device.type == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
        synchronize()
        start = time.perf_counter()
        if start_event is not None:
            start_event.record()
        for _ in range(iterations):
            model(x)
        if end_event is not None:
            end_event.record()
        synchronize()
        wall_ms = (time.perf_counter() - start) * 1000 / iterations
        gpu_ms = start_event.elapsed_time(end_event) / iterations if start_event is not None else None
        return wall_ms, gpu_ms

    with torch.inference_mode():
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 3, 32, 32, device=device)
            if masked is not None:
                torch.testing.assert_close(masked(x), pruned(x), rtol=1e-4, atol=1e-5)
            for model in models.values():
                for _ in range(warmup):
                    model(x)
            synchronize()
            samples = {name: [] for name in models}
            gpu_samples = {name: [] for name in models}
            for round_index in range(rounds):
                order = measurement_order(models, round_index)
                for name in order:
                    wall_ms, gpu_ms = measure(models[name], x)
                    samples[name].append(wall_ms)
                    if gpu_ms is not None:
                        gpu_samples[name].append(gpu_ms)
            batch = {'batch_size': batch_size}
            for name in models:
                median = statistics.median(samples[name])
                batch[name] = {
                    'median_ms_per_batch': median, 'images_per_second': batch_size * 1000 / median,
                    'round_ms_per_batch': samples[name],
                    'min_ms': min(samples[name]), 'max_ms': max(samples[name]),
                    'gpu_event_round_ms_per_batch': gpu_samples[name],
                }
            speedup = batch['original']['median_ms_per_batch'] / batch['pruned']['median_ms_per_batch']
            batch['speedup'] = speedup
            batch['faster_by_median'] = speedup > 1
            add_masked_ratios(batch)
            result['batches'].append(batch)
            print(f"[Benchmark] batch={batch_size} original={batch['original']['median_ms_per_batch']:.3f} ms "
                  f"pruned={batch['pruned']['median_ms_per_batch']:.3f} ms speedup={speedup:.3f}x", flush=True)
            if masked is not None:
                print(f"[Masked control] batch={batch_size} masked={batch['masked']['median_ms_per_batch']:.3f} ms "
                      f"compaction_speedup={batch['masked_to_pruned_speedup']:.3f}x", flush=True)
            if compile_requested:
                try:
                    compiled_batch = benchmark_compiled_pair(models, x, measure, warmup, rounds)
                    for name in models:
                        compiled_batch[name]['speedup_vs_eager'] = (
                            batch[name]['median_ms_per_batch'] / compiled_batch[name]['median_ms_per_batch'])
                    result['compiled']['batches'].append(compiled_batch)
                    print(f"[Compiled] batch={batch_size} pruning_speedup={compiled_batch['speedup']:.3f}x "
                          f"pruned_vs_eager={compiled_batch['pruned']['speedup_vs_eager']:.3f}x", flush=True)
                except Exception as exc:
                    result['compiled']['errors'].append({'batch_size': batch_size, 'error': f'{type(exc).__name__}: {exc}'})
                    print(f'[Compiled] FAILED batch={batch_size}: {type(exc).__name__}: {exc}', flush=True)
    if compile_requested:
        result['compiled']['status'] = ('success' if not result['compiled']['errors'] else
                                        'partial' if result['compiled']['batches'] else 'failed')
    result['masked_control']['outputs_verified'] = masked is not None
    print(f'[Benchmark] Parameter reduction: {100 * reduction:.2f}%; '
          f'at least 50% smaller: {reduction >= 0.5}', flush=True)
    Path(output_path).write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def measurement_order(models, round_index):
    names = list(models)
    if len(names) == 2:
        return names if round_index % 2 == 0 else names[::-1]
    shift = round_index % len(names)
    return names[shift:] + names[:shift]


def add_masked_ratios(batch):
    if 'masked' in batch:
        batch['original_to_masked_speedup'] = batch['original']['median_ms_per_batch'] / batch['masked']['median_ms_per_batch']
        batch['masked_to_pruned_speedup'] = batch['masked']['median_ms_per_batch'] / batch['pruned']['median_ms_per_batch']


def benchmark_compiled_pair(models, x, measure, warmup, rounds):
    """Compile fixed shapes, check outputs, warm up, then time both compiled models.

    Reset compiler guards per shape to avoid the recompile limit across the many
    architectures in the sweep. This function runs in the isolated experiment process.
    fullgraph=True rejects graph breaks instead of reporting eager fallback as compiled.
    """
    compiled = {}
    try:
        torch.compiler.reset()
        start = time.perf_counter()
        with torch.inference_mode():
            for name, model in models.items():
                compiled[name] = torch.compile(model, mode='reduce-overhead', fullgraph=True, dynamic=False)
                expected = model(x).clone()
                actual = compiled[name](x).clone()  # Triggers lazy compilation, outside timing.
                torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)
                for _ in range(max(warmup, 2)):
                    compiled[name](x)
            if x.device.type == 'cuda':
                torch.cuda.synchronize(x.device)
            setup_seconds = time.perf_counter() - start
            samples = {name: [] for name in models}
            gpu_samples = {name: [] for name in models}
            for round_index in range(rounds):
                order = measurement_order(models, round_index)
                for name in order:
                    wall_ms, gpu_ms = measure(compiled[name], x)
                    samples[name].append(wall_ms)
                    if gpu_ms is not None:
                        gpu_samples[name].append(gpu_ms)
        result = {'batch_size': x.shape[0], 'setup_seconds': setup_seconds,
                  'setup_includes': 'Lazy compilation, output check and warmup; excluded from inference latency.',
                  'warmup': max(warmup, 2), 'outputs_verified': True}
        for name in models:
            median = statistics.median(samples[name])
            result[name] = {'median_ms_per_batch': median, 'images_per_second': x.shape[0] * 1000 / median,
                            'round_ms_per_batch': samples[name], 'min_ms': min(samples[name]), 'max_ms': max(samples[name]),
                            'gpu_event_round_ms_per_batch': gpu_samples[name]}
        result['speedup'] = result['original']['median_ms_per_batch'] / result['pruned']['median_ms_per_batch']
        result['faster_by_median'] = result['speedup'] > 1
        add_masked_ratios(result)
        return result
    finally:
        compiled.clear()
        torch.compiler.reset()


def compiled_csv_fields(benchmark):
    fields = {}
    for batch in benchmark.get('compiled', {}).get('batches', []):
        size = batch['batch_size']
        fields[f'compiled_ms_batch_{size}'] = batch['pruned']['median_ms_per_batch']
        fields[f'compiled_pruning_speedup_batch_{size}'] = batch['speedup']
        fields[f'compile_gain_pruned_batch_{size}'] = batch['pruned']['speedup_vs_eager']
        if 'masked' in batch:
            fields[f'compiled_masked_ms_batch_{size}'] = batch['masked']['median_ms_per_batch']
            fields[f'compiled_masked_to_pruned_speedup_batch_{size}'] = batch['masked_to_pruned_speedup']
    return fields


COMPILED_CSV_COLUMNS = [f'{prefix}_{size}' for size in (1, 128, 1024)
                        for prefix in ('compiled_ms_batch', 'compiled_pruning_speedup_batch', 'compile_gain_pruned_batch',
                                       'compiled_masked_ms_batch', 'compiled_masked_to_pruned_speedup_batch')]


def masked_csv_fields(benchmark):
    fields = {}
    if 'masked' in benchmark.get('parameters', {}):
        fields['masked_parameters'] = benchmark['parameters']['masked']
    for batch in benchmark['batches']:
        if 'masked' in batch:
            size = batch['batch_size']
            fields[f'masked_ms_batch_{size}'] = batch['masked']['median_ms_per_batch']
            fields[f'masked_to_pruned_speedup_batch_{size}'] = batch['masked_to_pruned_speedup']
            fields[f'original_to_masked_speedup_batch_{size}'] = batch['original_to_masked_speedup']
    return fields


MASKED_CSV_COLUMNS = ['masked_parameters'] + [f'{prefix}_{size}' for size in (1, 128, 1024)
                     for prefix in ('masked_ms_batch', 'masked_to_pruned_speedup_batch', 'original_to_masked_speedup_batch')]


def aggregate_compiled_benchmarks(benchmarks):
    """Only aggregate batch sizes that succeeded for every training seed."""
    result = {'mode': 'reduce-overhead', 'batches': [], 'errors': [], 'status': 'skipped'}
    compiled = [b.get('compiled', {}) for b in benchmarks]
    if not any(c.get('status') in ('success', 'partial', 'failed') for c in compiled):
        return result
    for eager_batch in benchmarks[0]['batches']:
        size = eager_batch['batch_size']
        measured = [next((b for b in c.get('batches', []) if b['batch_size'] == size), None) for c in compiled]
        if any(b is None for b in measured):
            result['errors'].append({'batch_size': size, 'error': 'Compiled measurements missing for one or more seeds.'})
            continue
        batch = {'batch_size': size}
        names = ['original', 'pruned'] + (['masked'] if all('masked' in b for b in measured) else [])
        for name in names:
            values = [b[name]['median_ms_per_batch'] for b in measured]
            gains = [b[name]['speedup_vs_eager'] for b in measured]
            batch[name] = {'median_ms_per_batch': statistics.mean(values), 'per_seed_median_ms': values,
                           'std_ms': statistics.stdev(values) if len(values) > 1 else 0.0,
                           'speedup_vs_eager': statistics.mean(gains), 'per_seed_speedup_vs_eager': gains}
        ratios = [b['speedup'] for b in measured]
        batch['speedup'] = statistics.mean(ratios)
        batch['speedup_std'] = statistics.stdev(ratios) if len(ratios) > 1 else 0.0
        aggregate_masked_ratios(batch, measured)
        result['batches'].append(batch)
    result['status'] = 'success' if not result['errors'] else 'partial' if result['batches'] else 'failed'
    return result


def aggregate_masked_ratios(batch, measured):
    if 'masked' in batch:
        for key, numerator, denominator in (('masked_to_pruned_speedup', 'masked', 'pruned'),
                                             ('original_to_masked_speedup', 'original', 'masked')):
            ratios = [b[numerator]['median_ms_per_batch'] / b[denominator]['median_ms_per_batch'] for b in measured]
            batch[key] = statistics.mean(ratios)
            batch[key + '_std'] = statistics.stdev(ratios) if len(ratios) > 1 else 0.0



from dataclasses import dataclass, asdict, replace
import csv


@dataclass
class ExperimentConfig:
    seed: int = 123
    split_seed: int = 123
    data_root: str = '.'
    val_size: int = 5000
    levels: tuple = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85)
    baseline_epochs: int = 80
    finetune_epochs: int = 30
    patience: int = 6
    min_epochs: int = 8
    min_delta: float = 0.001  # accuracy fraction (0.1 percentage point)
    learning_rate: float = 0.001
    finetune_lr: float = 0.0003
    accuracy_tolerance: float = 0.01
    selection_batch: int = 128
    min_hidden: int = 4
    depth_removals: tuple = (1, 2)
    min_blocks: int = 1
    combined_width: float = 0.75

    def __post_init__(self):
        if not self.levels or any(not 0 < x < 1 for x in self.levels) or list(self.levels) != sorted(set(self.levels)):
            raise ValueError('Levels must be unique increasing fractions between 0 and 1')
        if min(self.val_size, self.baseline_epochs, self.finetune_epochs, self.patience, self.min_epochs, self.min_hidden) < 1:
            raise ValueError('Epoch counts, patience, validation size and minimum width must be positive')
        if self.min_epochs > min(self.baseline_epochs, self.finetune_epochs):
            raise ValueError('min_epochs must not exceed stage epoch limits')
        if self.min_delta < 0 or not 0 <= self.accuracy_tolerance <= 1 or min(self.learning_rate, self.finetune_lr) <= 0:
            raise ValueError('Invalid stopping, accuracy or learning-rate configuration')
        if self.selection_batch not in (1, 128, 1024):
            raise ValueError('selection_batch must be 1, 128 or 1024')
        if (self.min_blocks < 1 or not 0 < self.combined_width < 1 or
                list(self.depth_removals) != sorted(set(self.depth_removals)) or
                any(not isinstance(n, int) or n < 1 for n in self.depth_removals)):
            raise ValueError('Invalid block-removal configuration')


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    loss = correct = count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss += F.cross_entropy(logits, y, reduction='sum').item()
        correct += (logits.argmax(1) == y).sum().item()
        count += y.numel()
    if not count:
        raise ValueError('Evaluation dataset is empty')
    return {'loss': loss / count, 'accuracy': correct / count}


def fit_stage(model, train_loader, val_loader, device, config, max_epochs, lr, label):
    """Select and stop by validation accuracy; use loss only to break exact ties."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,
                                                        threshold=config.min_delta, threshold_mode='abs', min_lr=1e-6)
    metrics = evaluate(model, val_loader, device)
    best_loss, best_accuracy = metrics['loss'], metrics['accuracy']
    significant_accuracy = best_accuracy
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = bad_epochs = 0
    history = [{'epoch': 0, 'val_loss': metrics['loss'], 'val_accuracy': metrics['accuracy'], 'lr': lr}]
    for epoch in range(1, max_epochs + 1):
        train(model, optimizer, train_loader, device)
        metrics = evaluate(model, val_loader, device)
        history.append({'epoch': epoch, 'val_loss': metrics['loss'], 'val_accuracy': metrics['accuracy'],
                        'lr': optimizer.param_groups[0]['lr']})
        if (metrics['accuracy'] > best_accuracy or
                (metrics['accuracy'] == best_accuracy and metrics['loss'] < best_loss)):
            best_loss, best_accuracy, best_epoch = metrics['loss'], metrics['accuracy'], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if metrics['accuracy'] > significant_accuracy + config.min_delta:
            significant_accuracy, bad_epochs = metrics['accuracy'], 0
        else:
            bad_epochs += 1
        scheduler.step(metrics['accuracy'])
        print(f'[{label}] epoch={epoch} val_loss={metrics["loss"]:.4f} '
              f'val_accuracy={metrics["accuracy"]:.4f} lr={optimizer.param_groups[0]["lr"]:.6g}', flush=True)
        if epoch >= config.min_epochs and bad_epochs >= config.patience:
            break
    model.load_state_dict(best_state)
    return {'epochs_run': epoch, 'best_epoch': best_epoch, 'best_val_loss': best_loss,
            'best_val_accuracy': best_accuracy, 'monitor': 'validation_accuracy',
            'stopped_early': epoch < max_epochs, 'history': history,
            'restored_validation': evaluate(model, val_loader, device)}


def split_indices(size, val_size, seed):
    if not 0 < val_size < size:
        raise ValueError('Validation size must be smaller than the training dataset')
    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed)).tolist()
    return order[val_size:], order[:val_size]


def initial_masks(model):
    masks = []
    for block in model.mixer_blocks:
        mask = {}
        for name in ('token_mlp', 'channel_mlp'):
            mlp = getattr(block, name)
            mask[name + '.fc1'] = torch.ones_like(mlp.fc1.weight)
            mask[name + '.fc2'] = torch.ones_like(mlp.fc2.weight)
        masks.append(mask)
    return masks


def prune_to_level(model, original_dims, reduction, min_hidden=4, return_indices=False):
    """Rank redundant hidden neuron pairs using incoming AND outgoing correlations.

    Target widths refer to the original dense model. Greedily discard the member
    with smaller outgoing weight norm. The width floor can limit actual reduction.
    """
    masks = initial_masks(model)
    for block_index, block in enumerate(model.mixer_blocks):
        for kind, name in enumerate(('token_mlp', 'channel_mlp')):
            mlp = getattr(block, name)
            width = mlp.fc1.out_features
            keep = min(width, max(min_hidden, round(original_dims[block_index][kind] * (1 - reduction))))
            if keep == width:
                continue
            incoming = mlp.fc1.weight.detach().float().cpu().numpy()
            outgoing = mlp.fc2.weight.detach().float().cpu().numpy().T
            def correlation(vectors):
                centered = vectors - vectors.mean(axis=1, keepdims=True)
                norms = np.linalg.norm(centered, axis=1, keepdims=True)
                normalized = centered / np.maximum(norms, 1e-12)
                return np.abs(normalized @ normalized.T)
            score = (correlation(incoming) + correlation(outgoing)) / 2
            rows, cols = np.triu_indices(width, k=1)
            order = np.argsort(-score[rows, cols], kind='stable')
            alive = np.ones(width, dtype=bool)
            strength = np.linalg.norm(outgoing, axis=1)
            count = width
            for pair in order:
                i, j = rows[pair], cols[pair]
                if alive[i] and alive[j]:
                    removed = j if strength[j] <= strength[i] else i
                    alive[removed] = False
                    count -= 1
                    if count == keep:
                        break
            removed = torch.tensor(np.flatnonzero(~alive), device=mlp.fc1.weight.device)
            masks[block_index][name + '.fc1'][removed] = 0
            masks[block_index][name + '.fc2'][:, removed] = 0
    indices = [{name: (mask[name + '.fc1'].sum(dim=1) > 0).nonzero(as_tuple=True)[0].cpu().tolist()
                for name in ('token_mlp', 'channel_mlp')} for mask in masks]
    compact = compact_and_verify(model, masks, torch.randn(4, 3, 32, 32, device=next(model.parameters()).device))
    return (compact, indices) if return_indices else compact


def expand_compact_to_masked(compact, dense_template, kept_indices, kept_blocks=None):
    """Scatter the FINAL fine-tuned compact weights to their original neuron slots.

    This creates the same function using the original dense matrix dimensions.
    Missing neurons and their biases are zero, without adding runtime mask hooks.
    """
    kept_blocks = list(range(len(dense_template.mixer_blocks))) if kept_blocks is None else list(kept_blocks)
    if (len(kept_blocks) != len(compact.mixer_blocks) or len(kept_indices) != len(compact.mixer_blocks)
            or kept_blocks != sorted(set(kept_blocks))
            or any(i < 0 or i >= len(dense_template.mixer_blocks) for i in kept_blocks)):
        raise ValueError('Block counts and index mappings must match')
    masked = copy.deepcopy(dense_template).eval()
    masked.embedding.load_state_dict(compact.embedding.state_dict())
    masked.norm.load_state_dict(compact.norm.state_dict())
    masked.classifier.load_state_dict(compact.classifier.state_dict())
    with torch.no_grad():
        # Removed blocks stay present in this dense control but act as identity.
        for index, block in enumerate(masked.mixer_blocks):
            if index not in kept_blocks:
                for name in ('token_mlp', 'channel_mlp'):
                    mlp = getattr(block, name)
                    for fc in (mlp.fc1, mlp.fc2):
                        fc.weight.zero_()
                        if fc.bias is not None:
                            fc.bias.zero_()
        for source, block_id, mapping in zip(compact.mixer_blocks, kept_blocks, kept_indices):
            dest = masked.mixer_blocks[block_id]
            dest.norm1.load_state_dict(source.norm1.state_dict())
            dest.norm2.load_state_dict(source.norm2.state_dict())
            for name in ('token_mlp', 'channel_mlp'):
                small, large = getattr(source, name), getattr(dest, name)
                indices = mapping[name]
                if (len(indices) != small.fc1.out_features or len(set(indices)) != len(indices)
                        or any(not isinstance(i, int) or i < 0 or i >= large.fc1.out_features for i in indices)):
                    raise ValueError(f'Invalid kept-neuron indices for {name}')
                ids = torch.tensor(indices, device=large.fc1.weight.device, dtype=torch.long)
                large.fc1.weight.zero_()
                large.fc2.weight.zero_()
                large.fc1.weight[ids] = small.fc1.weight
                large.fc2.weight[:, ids] = small.fc2.weight
                if large.fc1.bias is not None:
                    large.fc1.bias.zero_()
                    large.fc1.bias[ids] = small.fc1.bias
                if large.fc2.bias is not None:
                    large.fc2.bias.copy_(small.fc2.bias)
    return masked


def remove_mixer_block(model, local_index, min_blocks=1):
    """Remove a complete token+channel Mixer block while retaining outer dimensions."""
    if len(model.mixer_blocks) <= min_blocks or min_blocks < 1:
        raise ValueError('Cannot remove another block below min_blocks')
    if not 0 <= local_index < len(model.mixer_blocks):
        raise ValueError('Invalid block index')
    candidate = copy.deepcopy(model).eval()
    candidate.mixer_blocks = nn.Sequential(*[block for i, block in enumerate(candidate.mixer_blocks) if i != local_index])
    return candidate


def choose_block_removal(model, kept_blocks, val_loader, device, min_blocks=1):
    """Greedily rank all single-block deletions using validation only."""
    if len(kept_blocks) != len(model.mixer_blocks):
        raise ValueError('Block mapping does not match model depth')
    if len(model.mixer_blocks) <= min_blocks:
        raise ValueError('No removable block remains')
    before = evaluate(model, val_loader, device)
    candidates = []
    for local_index, original_index in enumerate(kept_blocks):
        candidate = remove_mixer_block(model, local_index, min_blocks)
        metrics = evaluate(candidate, val_loader, device)
        candidates.append({'local_index': local_index, 'original_index': original_index, **metrics})
        del candidate
    selected = min(candidates, key=lambda row: (-row['accuracy'], row['loss'], row['original_index']))
    candidate = remove_mixer_block(model, selected['local_index'], min_blocks)
    # Independently verify physical deletion against bypassing the same block.
    bypass = copy.deepcopy(model).eval()
    bypass.mixer_blocks[selected['local_index']] = nn.Identity()
    example = torch.randn(4, 3, 32, 32, device=device)
    with torch.inference_mode():
        torch.testing.assert_close(candidate(example), bypass(example), rtol=1e-4, atol=1e-5)
    remaining = [i for i in kept_blocks if i != selected['original_index']]
    decision = {'validation_before': before, 'candidates': candidates, 'selected': selected,
                'physical_removal_verified': True}
    return candidate, remaining, decision


def model_config(model):
    return {'patch_size': model.patch_size, 'in_channels': model.in_channels,
            'embedding_dim': model.embedding.out_features, 'num_classes': model.classifier.out_features,
            'mixer_block_dims': [[b.token_mlp.fc1.in_features, b.channel_mlp.fc1.in_features,
                                  b.token_mlp.fc1.out_features, b.token_mlp.fc2.out_features,
                                  b.channel_mlp.fc1.out_features, b.channel_mlp.fc2.out_features]
                                 for b in model.mixer_blocks]}


def select_candidate(stages, tolerance, batch_size):
    """Use only validation accuracy, parameter count and measured inference time."""
    def latency(row):
        return next(b['pruned']['median_ms_per_batch'] for b in row['benchmark']['batches'] if b['batch_size'] == batch_size)
    best_accuracy = max(row['validation']['accuracy'] for row in stages)
    eligible = [row for row in stages if row['validation']['accuracy'] >= best_accuracy - tolerance - 1e-12]
    selected = min(eligible, key=lambda row: (latency(row), row['parameters']))
    pareto = []
    for row in stages:
        dominated = any(
            other['validation']['accuracy'] >= row['validation']['accuracy'] and
            other['parameters'] <= row['parameters'] and latency(other) <= latency(row) and
            (other['validation']['accuracy'] > row['validation']['accuracy'] or
             other['parameters'] < row['parameters'] or latency(other) < latency(row))
            for other in stages)
        if not dominated:
            pareto.append(row['stage'])
    return selected, pareto


def lottery_ticket_mixer_cycle(config=None, evaluate_test=True):
    config = config or ExperimentConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(), normalize])
    eval_transform = transforms.Compose([transforms.ToTensor(), normalize])
    print(f'[Data] CIFAR-10 mirror: {os.environ.get("MIXER_CIFAR10_URL") or DEFAULT_CIFAR10_URL}', flush=True)
    augmented = MirrorCIFAR10(config.data_root, train=True, download=True, transform=train_transform)
    validation_data = MirrorCIFAR10(config.data_root, train=True, download=False, transform=eval_transform)
    training_ids, validation_ids = split_indices(len(augmented), config.val_size, config.split_seed)
    Path('split_indices.json').write_text(json.dumps({'seed': config.split_seed, 'training': training_ids,
                                                     'validation': validation_ids}), encoding='utf-8')
    train_loader = DataLoader(torch.utils.data.Subset(augmented, training_ids), batch_size=128, shuffle=True,
                              generator=torch.Generator().manual_seed(config.seed), pin_memory=device.type == 'cuda')
    val_loader = DataLoader(torch.utils.data.Subset(validation_data, validation_ids), batch_size=128)
    original = MLPMixer().to(device)
    if config.min_blocks > len(original.mixer_blocks):
        raise ValueError('min_blocks exceeds original model depth')
    original_dims = [(b.token_mlp.fc1.out_features, b.channel_mlp.fc1.out_features) for b in original.mixer_blocks]
    dense_training = fit_stage(original, train_loader, val_loader, device, config, config.baseline_epochs,
                               config.learning_rate, 'dense')
    torch.save(original.state_dict(), 'original_mixer_model.pt')
    model = original
    stages = []
    summary = {'configuration': asdict(config), 'stages': stages,
               'selection_rule': 'Fastest at selection_batch within accuracy_tolerance of best validation accuracy; test unused.'}
    def write_summary():
        Path('experiment_results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    def record_stage(model, kept_indices, kept_blocks, target, family, training, decisions=None):
        index = len(stages)
        filename = f'stage_{index:02d}.pt'
        torch.save(model.state_dict(), filename)
        masked = expand_compact_to_masked(model, original, kept_indices, kept_blocks)
        benchmark = benchmark_models(original, model, masked=masked)
        masked_validation = evaluate(masked, val_loader, device)
        del masked
        Path(f'benchmark_stage_{index:02d}.json').write_text(json.dumps(benchmark, indent=2), encoding='utf-8')
        removed = len(original.mixer_blocks) - len(model.mixer_blocks)
        row = {'stage': index, 'family': family, 'candidate_key': f'{family}:width={target:.4f}:removed={removed}',
               'depth': len(model.mixer_blocks), 'blocks_removed': removed,
               'target_reduction': target, 'parameters': sum(p.numel() for p in model.parameters()),
               'actual_reduction': benchmark['parameter_reduction_fraction'], 'validation': training['restored_validation'],
               'training': training, 'checkpoint': filename, 'model_config': model_config(model), 'benchmark': benchmark,
               'kept_indices': kept_indices, 'kept_blocks': list(kept_blocks), 'masked_validation': masked_validation,
               'depth_decisions': copy.deepcopy(decisions or [])}
        stages.append(row)
        print(f'[Stage] family={family} width_target={target:.0%} depth={row["depth"]} '
              f'actual_reduction={row["actual_reduction"]:.2%} val_accuracy={row["validation"]["accuracy"]:.4f}', flush=True)
        write_summary()
        return row

    full_blocks = list(range(len(original.mixer_blocks)))
    max_removals = min(max(config.depth_removals, default=0), len(full_blocks) - config.min_blocks)
    targets = sorted(set(config.levels) | ({config.combined_width} if max_removals > 0 else set()))
    width_stages = {}
    for target in (0.0,) + tuple(targets):
        if target:
            model, kept_indices = prune_to_level(copy.deepcopy(original), original_dims, target,
                                                 config.min_hidden, return_indices=True)
            training = fit_stage(model, train_loader, val_loader, device, config, config.finetune_epochs,
                                 config.finetune_lr, f'prune-{target:.0%}')
        else:
            model, training = original, dense_training
            kept_indices = [{name: list(range(getattr(block, name).fc1.out_features))
                             for name in ('token_mlp', 'channel_mlp')} for block in original.mixer_blocks]
        width_stages[target] = record_stage(model, kept_indices, full_blocks, target,
                                            'width' if target else 'dense', training)

    # Separate greedy depth-only and width+depth branches; never inspect test data.
    if max_removals > 0:
        for family, width in (('depth', 0.0), ('combined', config.combined_width)):
            base = width_stages[width]
            model = MLPMixer(**base['model_config']).to(device)
            model.load_state_dict(torch.load(base['checkpoint'], map_location=device, weights_only=True))
            kept_blocks, indices = list(full_blocks), copy.deepcopy(base['kept_indices'])
            decisions = []
            for removed in range(1, max_removals + 1):
                model, kept_blocks, decision = choose_block_removal(model, kept_blocks, val_loader, device, config.min_blocks)
                indices.pop(decision['selected']['local_index'])
                decisions.append(decision)
                training = fit_stage(model, train_loader, val_loader, device, config, config.finetune_epochs,
                                     config.finetune_lr, f'{family}-remove-{removed}')
                if removed in config.depth_removals:
                    record_stage(model, indices, kept_blocks, width, family, training, decisions)
    selected, pareto = select_candidate(stages, config.accuracy_tolerance, config.selection_batch)
    summary.update(selected_stage=selected['stage'], pareto_stages=pareto)
    # Freeze selection before constructing or evaluating the held-out test dataset.
    write_summary()
    chosen = MLPMixer(**selected['model_config']).to(device)
    chosen.load_state_dict(torch.load(selected['checkpoint'], map_location=device, weights_only=True))
    torch.save(chosen.state_dict(), 'pruned_mixer_model.pt')
    result = copy.deepcopy(selected['benchmark'])
    result.update(selected_stage=selected['stage'],
                  training={'note': 'Width branches start from the same trained dense baseline. Depth-only and '
                                    'combined branches greedily remove blocks with sequential fine-tuning. '
                                    'Training budgets differ; held-out test never guides deletion or selection.'})
    if evaluate_test:
        test_loader = DataLoader(MirrorCIFAR10(config.data_root, train=False, download=True, transform=eval_transform), batch_size=128)
        dense_test = evaluate(original, test_loader, device)
        selected_test = evaluate(chosen, test_loader, device)
        masked = expand_compact_to_masked(chosen, original, selected['kept_indices'], selected.get('kept_blocks'))
        masked_test = evaluate(masked, test_loader, device)
        torch.save(masked.state_dict(), 'pruned_masked_model.pt')
        summary['held_out_test'] = {'dense': dense_test, 'selected': selected_test, 'masked': masked_test}
        del masked
        result['accuracy'] = {'original': dense_test['accuracy'], 'pruned': selected_test['accuracy'],
                             'change_percentage_points': 100 * (selected_test['accuracy'] - dense_test['accuracy'])}
    Path('benchmark_results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    write_summary()
    with Path('stage_summary.csv').open('w', newline='', encoding='utf-8') as handle:
        columns = ['stage', 'family', 'depth', 'blocks_removed', 'target_reduction', 'actual_reduction', 'parameters', 'val_accuracy', 'val_loss',
                   'ms_batch_1', 'ms_batch_128', 'ms_batch_1024', 'selected', 'pareto'] + COMPILED_CSV_COLUMNS + MASKED_CSV_COLUMNS
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in stages:
            times = {f'ms_batch_{b["batch_size"]}': b['pruned']['median_ms_per_batch'] for b in row['benchmark']['batches']}
            writer.writerow(dict(stage=row['stage'], family=row['family'], depth=row['depth'], blocks_removed=row['blocks_removed'],
                                 target_reduction=row['target_reduction'], actual_reduction=row['actual_reduction'],
                                 parameters=row['parameters'], val_accuracy=row['validation']['accuracy'], val_loss=row['validation']['loss'],
                                 selected=row['stage'] == selected['stage'], pareto=row['stage'] in pareto,
                                 **times, **compiled_csv_fields(row['benchmark']), **masked_csv_fields(row['benchmark'])))
    print(f'[Selected] stage={selected["stage"]} parameters={selected["parameters"]} '
          f'validation={selected["validation"]["accuracy"]:.4f}', flush=True)
    return summary


def aggregate_repeats(runs, tolerance, selection_batch):
    """Compare means across seeds. Uncertainty is variation across training seeds."""
    if not runs or any(len(run['stages']) != len(runs[0]['stages']) for run in runs):
        raise ValueError('Repeated runs must contain matching stages')
    stages = []
    for index in range(len(runs[0]['stages'])):
        rows = [run['stages'][index] for run in runs]
        if len({row['target_reduction'] for row in rows}) != 1:
            raise ValueError('Repeated runs have different pruning targets')
        if len({row.get('candidate_key', str(row['target_reduction'])) for row in rows}) != 1:
            raise ValueError('Repeated runs have different width/depth candidates')
        accuracies = [row['validation']['accuracy'] for row in rows]
        batches = []
        for batch_index in range(len(rows[0]['benchmark']['batches'])):
            measured = [row['benchmark']['batches'][batch_index] for row in rows]
            batch = {'batch_size': measured[0]['batch_size']}
            names = ['original', 'pruned'] + (['masked'] if all('masked' in b for b in measured) else [])
            for name in names:
                samples = [b[name]['median_ms_per_batch'] for b in measured]
                batch[name] = {'median_ms_per_batch': statistics.mean(samples),
                               'per_seed_median_ms': samples,
                               'std_ms': statistics.stdev(samples) if len(samples) > 1 else 0.0}
            paired_speedups = [b['original']['median_ms_per_batch'] / b['pruned']['median_ms_per_batch'] for b in measured]
            batch['speedup'] = statistics.mean(paired_speedups)
            batch['speedup_std'] = statistics.stdev(paired_speedups) if len(paired_speedups) > 1 else 0.0
            aggregate_masked_ratios(batch, measured)
            batches.append(batch)
        stages.append({'stage': index, 'target_reduction': rows[0]['target_reduction'],
                       'family': rows[0].get('family', 'width'), 'depth': rows[0].get('depth'),
                       'blocks_removed': rows[0].get('blocks_removed', 0),
                       'candidate_key': rows[0].get('candidate_key'),
                       'per_seed_kept_blocks': [row.get('kept_blocks') for row in rows],
                       'parameters': rows[0]['parameters'], 'actual_reduction': rows[0]['actual_reduction'],
                       'validation': {'accuracy': statistics.mean(accuracies),
                                      'loss': statistics.mean(row['validation']['loss'] for row in rows)},
                       'validation_accuracy_std': statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
                       'per_seed_validation_accuracy': accuracies,
                       'benchmark': {'aggregation': 'Mean of per-seed latency medians; speedup is mean paired speedup.',
                                     'batches': batches,
                                     'parameters': rows[0]['benchmark'].get('parameters', {}),
                                     'compiled': aggregate_compiled_benchmarks([row['benchmark'] for row in rows])}})
    selected, pareto = select_candidate(stages, tolerance, selection_batch)
    return {'seeds': [run['configuration']['seed'] for run in runs], 'stages': stages,
            'selected_stage': selected['stage'], 'pareto_stages': pareto,
            'selection_rule': 'Mean validation accuracy and mean latency; held-out test unused.',
            'uncertainty_note': 'Sample standard deviations across seeds, not confidence intervals; '
                                'all runs share one fixed train/validation split.'}


def repeat_experiments(config, seeds=(123, 456, 789)):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Provide distinct training seeds')
    root = Path.cwd()
    shared_data = str(Path(config.data_root).resolve())
    runs = []
    for seed in seeds:
        directory = root / f'seed_{seed}'
        directory.mkdir(parents=True, exist_ok=True)
        print(f'[Repeat] seed={seed} split_seed={config.split_seed}', flush=True)
        try:
            os.chdir(directory)
            runs.append(lottery_ticket_mixer_cycle(replace(config, seed=seed, data_root=shared_data), evaluate_test=False))
        finally:
            os.chdir(root)
        Path('repeat_progress.json').write_text(json.dumps({'completed_seeds': [r['configuration']['seed'] for r in runs]}),
                                               encoding='utf-8')
    summary = aggregate_repeats(runs, config.accuracy_tolerance, config.selection_batch)
    summary['configuration'] = asdict(config)
    summary['status'] = 'selection_frozen_before_test'
    Path('repeat_results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    # Only now evaluate a common selected size on the held-out test set for every seed.
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    test_loader = DataLoader(MirrorCIFAR10(shared_data, train=False, download=True, transform=transform), batch_size=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tests = []
    for run, seed in zip(runs, seeds):
        directory = root / f'seed_{seed}'
        record = {'seed': seed}
        for name, index in (('dense', 0), ('selected', summary['selected_stage'])):
            stage = run['stages'][index]
            model = MLPMixer(**stage['model_config']).to(device)
            model.load_state_dict(torch.load(directory / stage['checkpoint'], map_location=device, weights_only=True))
            record[name] = evaluate(model, test_loader, device)
            if name == 'selected':
                torch.save(model.state_dict(), directory / 'repeat_selected_model.pt')
                dense_template = MLPMixer(**run['stages'][0]['model_config']).to(device)
                masked = expand_compact_to_masked(model, dense_template, stage['kept_indices'], stage.get('kept_blocks'))
                record['masked'] = evaluate(masked, test_loader, device)
                torch.save(masked.state_dict(), directory / 'repeat_selected_masked_model.pt')
                del masked, dense_template
            del model
        tests.append(record)
    summary['held_out_test'] = tests
    summary['status'] = 'success'
    summary['test_accuracy_mean'] = {name: statistics.mean(t[name]['accuracy'] for t in tests) for name in ('dense', 'selected', 'masked')}
    Path('repeat_results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    Path('benchmark_results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with Path('repeat_summary.csv').open('w', newline='', encoding='utf-8') as handle:
        fields = ['family', 'depth', 'blocks_removed', 'target_reduction', 'actual_reduction', 'parameters', 'val_accuracy_mean', 'val_accuracy_std',
                  'speedup_batch_1', 'speedup_batch_128', 'speedup_batch_1024', 'selected', 'pareto'] + COMPILED_CSV_COLUMNS + MASKED_CSV_COLUMNS
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary['stages']:
            writer.writerow(dict(family=row['family'], depth=row['depth'], blocks_removed=row['blocks_removed'],
                                 target_reduction=row['target_reduction'], actual_reduction=row['actual_reduction'],
                                 parameters=row['parameters'], val_accuracy_mean=row['validation']['accuracy'],
                                 val_accuracy_std=row['validation_accuracy_std'], selected=row['stage'] == summary['selected_stage'],
                                 pareto=row['stage'] in summary['pareto_stages'],
                                 **{f'speedup_batch_{b["batch_size"]}': b['speedup'] for b in row['benchmark']['batches']},
                                 **compiled_csv_fields(row['benchmark']), **masked_csv_fields(row['benchmark'])))
    print(f'[Repeated result] selected_stage={summary["selected_stage"]} test_means={summary["test_accuracy_mean"]}', flush=True)
    return summary


if __name__ == '__main__':
    config = ExperimentConfig(
        data_root=os.environ.get('MIXER_DATA_ROOT', './data'),
        levels=tuple(float(x) for x in os.environ.get('MIXER_LEVELS', '0.60,0.65,0.70,0.75,0.80,0.85').split(',')),
        baseline_epochs=int(os.environ.get('MIXER_BASELINE_EPOCHS', '80')),
        finetune_epochs=int(os.environ.get('MIXER_FINETUNE_EPOCHS', '30')),
        patience=int(os.environ.get('MIXER_PATIENCE', '6')),
        depth_removals=tuple(int(x) for x in os.environ.get('MIXER_DEPTH_REMOVALS', '1,2').split(',') if x.strip()),
        min_blocks=int(os.environ.get('MIXER_MIN_BLOCKS', '1')),
        combined_width=float(os.environ.get('MIXER_COMBINED_WIDTH', '0.75')),
    )
    repeat_experiments(config, tuple(int(x) for x in os.environ.get('MIXER_SEEDS', '123,456,789').split(',')))
