import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import copy
import matplotlib.pyplot as plt
import time
from models import MLPMixer, MixerBlock  # <-- Import MLP from models.py

# ==== 2. Tanítás és tesztelés ====
def train(model, optimizer, dataloader, device):
    model.train()
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

def test(model, dataloader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return correct / len(dataloader.dataset)

def correlation_mask(weight_tensor: torch.Tensor, global_mask=None, relative_margin=0.15, verbose=True, layer_name="layer", axis=0):
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

def prune_mixer_model(model, global_masks, relative_margin=0.15, verbose=True):
    """
    Korrelációalapú pruning minden MixerBlock token_mlp és channel_mlp részén.
    Csak az fc1 rétegekben történik korrelációszámítás.
    Az fc2 rétegek maszkolása az fc1 kiesett kimenetei alapján történik.
    """
    new_global_masks = []

    for i, block in enumerate(model.mixer_blocks):
        device = block.token_mlp.fc1.weight.device

        # === TOKEN MLP ===
        # fc1: korrelációs pruning
        mask_token_fc1 = correlation_mask(
            block.token_mlp.fc1.weight,
            global_mask=global_masks[i]['fc1'],
            relative_margin=relative_margin,
            verbose=verbose,
            layer_name=f"token_mlp_fc1_block_{i}"
        )

        # fc2 bemenete = fc1 output → propagált
        mask_token_fc2 = global_masks[i]['fc2'] * propagate_mask(mask_token_fc1, block.token_mlp.fc2.weight).to(device)

        # === CHANNEL MLP ===
        # fc1: korrelációs pruning
        mask_channel_fc1 = correlation_mask(
            block.channel_mlp.fc1.weight,
            global_mask=global_masks[i]['channel_fc1'],
            relative_margin=relative_margin,
            verbose=verbose,
            layer_name=f"channel_mlp_fc1_block_{i}"
        )

        # fc2 bemenete = fc1 output → propagált
        mask_channel_fc2 = global_masks[i]['channel_fc2'] * propagate_mask(mask_channel_fc1, block.channel_mlp.fc2.weight).to(device)

        # === APPLY ALL MASKS ===
        with torch.no_grad():
            block.token_mlp.fc1.weight *= mask_token_fc1
            block.token_mlp.fc2.weight *= mask_token_fc2
            block.channel_mlp.fc1.weight *= mask_channel_fc1
            block.channel_mlp.fc2.weight *= mask_channel_fc2

        # === UPDATE GLOBAL MASKS ===
        new_global_masks.append({
            'fc1': global_masks[i]['fc1'] * mask_token_fc1,
            'fc2': global_masks[i]['fc2'] * mask_token_fc2,
            'channel_fc1': global_masks[i]['channel_fc1'] * mask_channel_fc1,
            'channel_fc2': global_masks[i]['channel_fc2'] * mask_channel_fc2
        })

    return new_global_masks


def build_fully_pruned_mixer_model(model, global_masks):
    """
    Új MLPMixer modellt épít a pruning maszkok alapján, megtartva a neuronok sorrendjét.
    Csak a token_mlp.fc1 és channel_mlp.fc1 output dimenzióit csökkentjük.
    """
    device = next(model.parameters()).device
    new_blocks = []

    for i, block in enumerate(model.mixer_blocks):
        # === Maszkok ===
        token_fc1_mask = global_masks[i]['fc1']  # [out, in]
        token_fc2_mask = global_masks[i]['fc2']  # [out, in]
        channel_fc1_mask = global_masks[i]['channel_fc1']
        channel_fc2_mask = global_masks[i]['channel_fc2']

        # === Aktív neuronindexek (megőrizzük sorrendet) ===
        token_fc1_rows = (token_fc1_mask.sum(dim=1) > 0).nonzero(as_tuple=True)[0]
        channel_fc1_rows = (channel_fc1_mask.sum(dim=1) > 0).nonzero(as_tuple=True)[0]

        # Dimenziók
        num_patches = block.token_mlp.fc1.in_features  # bemenet mérete a token_mlp-hez (pl. 49 patch)
        dim = block.channel_mlp.fc1.in_features        # fix: minden blokk ugyanazzal a dim-mel dolgozik
        token_dim = len(token_fc1_rows)
        channel_dim = len(channel_fc1_rows)

        # === Új blokk létrehozása ===
        new_block = MixerBlock(
            num_patches=num_patches,
            dim=dim,
            token_dim=token_dim,
            channel_dim=channel_dim
        ).to(device)

        # === Súlyok másolása ===
        with torch.no_grad():
            # TOKEN MLP
            new_block.token_mlp.fc1.weight.copy_(block.token_mlp.fc1.weight[token_fc1_rows])
            new_block.token_mlp.fc2.weight.copy_(block.token_mlp.fc2.weight[:, token_fc1_rows])

            # CHANNEL MLP
            new_block.channel_mlp.fc1.weight.copy_(block.channel_mlp.fc1.weight[channel_fc1_rows])
            new_block.channel_mlp.fc2.weight.copy_(block.channel_mlp.fc2.weight[:, channel_fc1_rows])
            

        # Normák
        new_block.norm1.load_state_dict(block.norm1.state_dict())
        new_block.norm2.load_state_dict(block.norm2.state_dict())

        new_blocks.append(new_block)

    # === Új modell összeállítása ===
    pruned_mixer = MLPMixer(
        image_size=int(model.patch_size * (model.mixer_blocks[0].token_mlp.fc1.in_features) ** 0.5),
        patch_size=model.patch_size,
        dim=model.embedding.out_features,
        depth=len(new_blocks),
        num_classes=model.classifier.out_features,
        token_dim=token_dim,      # csak formálisan kell, az új blokkokban külön szerepel
        channel_dim=channel_dim
    ).to(device)

    pruned_mixer.mixer_blocks = nn.Sequential(*new_blocks)

    # === Embedding, classifier, norm rétegek másolása ===
    pruned_mixer.embedding.load_state_dict(model.embedding.state_dict())
    pruned_mixer.norm.load_state_dict(model.norm.state_dict())
    pruned_mixer.classifier.load_state_dict(model.classifier.state_dict())

    return pruned_mixer


# ==== 4. LTH ciklus ====
def lottery_ticket_mixer_cycle(device, prune_steps=9, relative_margin=0.15):
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, pin_memory=True)

    model = MLPMixer().to(device)
    print(model)
    initial_state = copy.deepcopy(model.state_dict())

    # Globális maszk inicializálása minden blokkra
    global_masks = []
    for block in model.mixer_blocks:
        m = {
            'fc1': torch.ones_like(block.token_mlp.fc1.weight),
            'fc2': torch.ones_like(block.token_mlp.fc2.weight),
            'channel_fc1': torch.ones_like(block.channel_mlp.fc1.weight),
            'channel_fc2': torch.ones_like(block.channel_mlp.fc2.weight),
        }
        global_masks.append(m)

    for step in range(prune_steps):
        print(f"\n=== MIXER PRUNING STEP {step + 1} ===")

        model = MLPMixer().to(device)
        model.load_state_dict(copy.deepcopy(initial_state))
        optimizer = torch.optim.Adam(model.parameters())

        # Maszkolás alkalmazása a súlyokra
        with torch.no_grad():
            for i, block in enumerate(model.mixer_blocks):
                block.token_mlp.fc1.weight *= global_masks[i]['fc1'].to(device)
                block.token_mlp.fc2.weight *= global_masks[i]['fc2'].to(device)
                block.channel_mlp.fc1.weight *= global_masks[i]['channel_fc1'].to(device)
                block.channel_mlp.fc2.weight *= global_masks[i]['channel_fc2'].to(device)

        for epoch in range(3):
            train(model, optimizer, train_loader, device)
            start = time.time()
            acc = test(model, test_loader, device)
            elapsed = time.time() - start
            print(f"Epoch {epoch + 1}: accuracy={acc:.4f} - {elapsed:.4f} seconds")

        # Mentés az utolsó iterációban
        if step == prune_steps - 1:
            torch.save(model.state_dict(), "mixer_pruned_model.pt")
            print("Final pruned Mixer model saved as 'mixer_pruned_model.pt'")
            
            pruned_mixer = build_fully_pruned_mixer_model(model, global_masks)
            start = time.time()
            acc = test(pruned_mixer, test_loader, device)
            elapsed = time.time() - start
            torch.save(pruned_mixer.state_dict(), "pruned_mixer_model.pt")
            
            print(f"[Eval] pruned modell - Inference time on test set: {elapsed:.4f} seconds - accuracy={acc:.4f}")
            print(pruned_mixer)

            break

        # Új maszkolás
        global_masks = prune_mixer_model(model, global_masks, relative_margin=relative_margin)
        
        # Log pruning statistics
        for i, block in enumerate(model.mixer_blocks):
            for name, weight in [('token_mlp.fc1', block.token_mlp.fc1.weight), ('token_mlp.fc2', block.token_mlp.fc2.weight), 
                                 ('channel_mlp.fc1', block.channel_mlp.fc1.weight), ('channel_mlp.fc2', block.channel_mlp.fc2.weight)]:
                num_zero = torch.sum(weight == 0).item()
                total = weight.numel()
                sparsity = 100.0 * num_zero / total
                print(f"[Pruning Info] Block {i} {name} nullázott súlyok: {num_zero}/{total} ({sparsity:.2f}%)")

# === Futtatás ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lottery_ticket_mixer_cycle(device)