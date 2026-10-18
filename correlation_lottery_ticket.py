import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import copy
import matplotlib.pyplot as plt
import time

# --- Classic MLP model ---
class MLP(nn.Module):
    def __init__(self, in_features=784, hidden1=300, hidden2=100, out_features=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, out_features)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

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

def build_fully_pruned_torchscript_model(original_model, fc1_mask, fc2_mask, fc3_mask, save_path="mnist_fully_pruned_model.pt"):
    """
    Létrehoz egy új MLP modellt a megmaradt fc1, fc2 és fc3 neuronokkal, majd TorchScript-ként exportálja.
    A fc3 rétegben mindig biztosítja, hogy 10 kimeneti neuron (logit) megmaradjon.
    """
    device = next(original_model.parameters()).device
    
    example_input = torch.rand(1, 1, 28, 28).to(device)
    traced = torch.jit.trace(original_model, example_input)
    traced.save("original_model.pt")
    
    # --- 1. Aktív neuron indexek
    active_fc1_cols = torch.any(fc1_mask != 0, dim=0).nonzero(as_tuple=True)[0]  # bemenet → fc1 input
    active_fc2_cols = torch.any(fc2_mask != 0, dim=0).nonzero(as_tuple=True)[0]  # fc2 input ← fc1 output
    active_fc3_cols = torch.any(fc3_mask != 0, dim=0).nonzero(as_tuple=True)[0]  # fc3 input ← fc2 output

    active_fc1_rows = torch.any(fc1_mask != 0, dim=1).nonzero(as_tuple=True)[0]  # fc1 output
    active_fc2_rows = torch.any(fc2_mask != 0, dim=1).nonzero(as_tuple=True)[0]  # fc2 output
    active_fc3_rows = torch.any(fc3_mask != 0, dim=1).nonzero(as_tuple=True)[0]  # fc3 output (logitek)

    # --- 1.b Garantáljuk a 10 kimenetet (0–9 osztály)
    if len(active_fc3_rows) < 10:
        print(f"[Warning] fc3 kimeneti neuronok száma = {len(active_fc3_rows)} < 10 → kiegészítjük.")
        all_rows = torch.arange(10, device=fc3_mask.device)
        active_fc3_rows = all_rows

    # --- 2. Új modell méretezése
    in_fc1 = 784
    out_fc1 = len(active_fc1_rows)
    out_fc2 = len(active_fc2_rows)
    out_fc3 = 10  # végső logitok

    print(f"[Debug] Prunolt modell méretei: fc1: {in_fc1}x{out_fc1}, fc2: {out_fc1}x{out_fc2}, fc3: {out_fc2}x{out_fc3}")
    
    pruned_model = MLP(in_features=in_fc1, hidden1=out_fc1, hidden2=out_fc2, out_features=out_fc3).to(device)

    with torch.no_grad():
        pruned_model.fc1.weight.copy_(original_model.fc1.weight[active_fc1_rows])
        pruned_model.fc1.bias.copy_(original_model.fc1.bias[active_fc1_rows])

        # fc2: csak az aktív output neuronok (sorok) és bemenetek (oszlopok)
        pruned_model.fc2.weight.copy_(original_model.fc2.weight[active_fc2_rows][:, active_fc1_rows])
        pruned_model.fc2.bias.copy_(original_model.fc2.bias[active_fc2_rows])

        # fc3: kimeneti logitek és aktív bemenetek
        pruned_model.fc3.weight.copy_(original_model.fc3.weight[active_fc3_rows][:, active_fc2_rows])
        pruned_model.fc3.bias.copy_(original_model.fc3.bias[active_fc3_rows])

    pruned_model.eval()
    dummy_input = torch.randn(1, 1, 28, 28).to(device)  # MNIST-re például
    # ONNX exportálás
    torch.onnx.export(
        pruned_model,
        dummy_input,
        "pruned_model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=20
    )
    # Export TorchScript
    example_input = torch.rand(1, 1, 28, 28).to(device)
    traced = torch.jit.trace(pruned_model, example_input)
    traced.save(save_path)
    print(f"[TorchScript] Prunolt modell elmentve: {save_path}")
    return traced

# ==== 3. Korrelációs maszkolás ====
def correlation_mask(weight_tensor: torch.Tensor, global_mask=None, relative_margin=0.15, verbose=True, layer_name="layer"):
    W = weight_tensor.detach().cpu().numpy()

    if global_mask is not None:
        W = W * global_mask.cpu().numpy()

    row_norms = np.linalg.norm(W, axis=1)
    nonzero_rows = np.where(row_norms > 1e-6)[0]

    if len(nonzero_rows) < 2:
        if verbose:
            print(f"  [Debug] {layer_name}: kevés nem-nulla neuron maradt, nincs értelme korrelációt számolni.")
        return torch.ones_like(weight_tensor)

    w_filtered = W[nonzero_rows]
    corr_matrix = np.corrcoef(w_filtered)
    np.fill_diagonal(corr_matrix, 0.0)

    max_corr = np.nanmax(np.abs(corr_matrix))
    threshold = max(max_corr - relative_margin, 0.0)

    if verbose:
        print(f"  [Debug] {layer_name}: max(abs(corr)) = {max_corr:.4f}, dynamic threshold = {threshold:.4f}")

    mask = np.ones_like(W)
    already_pruned = set()

    for i in range(len(nonzero_rows)):
        for j in range(i + 1, len(nonzero_rows)):
            if abs(corr_matrix[i, j]) > threshold:
                j_idx = int(nonzero_rows[j])
                if j_idx not in already_pruned:
                    mask[j_idx, :] = 0.0
                    already_pruned.add(j_idx)

    if verbose:
        print(f"  [Debug] {layer_name}: neuronok nullázva (indexek): {sorted(list(already_pruned))}")

    full_mask = torch.ones_like(weight_tensor)
    full_mask *= torch.tensor(mask, dtype=torch.float32, device=full_mask.device)

    return full_mask

def propagate_mask(prev_mask: torch.Tensor, next_weight: torch.Tensor) -> torch.Tensor:
    row_sums = prev_mask.sum(dim=1)
    zero_rows = (row_sums == 0).nonzero(as_tuple=False).squeeze()
    next_mask = torch.ones_like(next_weight)
    if zero_rows.numel() > 0:
        next_mask[:, zero_rows] = 0.0
    return next_mask

# ==== 4. LTH ciklus ====
def lottery_ticket_cycle(device, prune_steps=7):
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    original_model = MLP().to(device)
    initial_state = copy.deepcopy(original_model.state_dict())

    # Generalize for any prunable layers
    prunable_layers = ['fc1', 'fc2', 'fc3']
    layer_shapes = {name: getattr(original_model, name).weight.shape for name in prunable_layers}
    global_masks = {name: torch.ones(shape, dtype=torch.float32) for name, shape in layer_shapes.items()}

    for step in range(prune_steps):
        print(f"\n=== PRUNING STEP {step+1} ===")
        model = MLP().to(device)
        model.load_state_dict(copy.deepcopy(initial_state))
        optimizer = torch.optim.Adam(model.parameters())

        # Apply cumulative mask BEFORE training
        with torch.no_grad():
            for lname in prunable_layers:
                layer = getattr(model, lname)
                weight_mask = global_masks[lname].to(device)
                layer.weight *= weight_mask

                # Bias nullázása, ha a súlysor 0
                bias = layer.bias
                weight = layer.weight
                for i in range(weight.shape[0]):
                    if torch.all(weight[i] == 0):
                        bias[i] = 0.0

        for epoch in range(3):
            train(model, optimizer, train_loader, device)
            acc = test(model, test_loader, device)
            print(f"Epoch {epoch+1}: accuracy={acc:.4f}")

        # Save model after each iteration (overwrite)
        torch.save(model.state_dict(), "pruned_model.pt")
        if step == prune_steps - 1:
            start_time = time.time()
            test(model, test_loader, device)
            elapsed = time.time() - start_time
            print(f"Test execution time of original model: {elapsed:.4f} seconds")
            build_fully_pruned_torchscript_model(model, global_masks["fc1"], global_masks["fc2"], global_masks["fc3"], save_path="pruned_model.pt")
            print("Final model saved as 'pruned_model.pt'.")
            exit()

        # Create new masks based on current state
        new_masks = {}

        # fc1 maszkolása önállóan
        new_masks["fc1"] = correlation_mask(
            model.fc1.weight,
            global_mask=global_masks["fc1"],
            relative_margin=0.15,
            layer_name="fc1"
        )

        # fc2 maszkolása, figyelembe véve a fc1 maszkolás hatását
        fc2_input_mask = propagate_mask(new_masks["fc1"], model.fc2.weight)
        combined_fc2_mask = global_masks["fc2"] * fc2_input_mask.to(global_masks["fc2"].device)
        new_masks["fc2"] = correlation_mask(
            model.fc2.weight,
            global_mask=combined_fc2_mask,
            relative_margin=0.15,
            layer_name="fc2"
        )

        # fc3 maszkolása, figyelembe véve a fc2 maszkolás hatását
        fc3_input_mask = propagate_mask(new_masks["fc2"], model.fc3.weight)
        combined_fc3_mask = global_masks["fc3"] * fc3_input_mask.to(global_masks["fc3"].device)
        new_masks["fc3"] = correlation_mask(
            model.fc3.weight,
            global_mask=combined_fc3_mask,
            relative_margin=0.15,
            layer_name="fc3"
        )

        # Update global masks
        for lname in prunable_layers:
            global_masks[lname] = global_masks[lname] * new_masks[lname].to(global_masks[lname].device)

        # Apply new cumulative mask
        with torch.no_grad():
            for lname in prunable_layers:
                getattr(model, lname).weight *= global_masks[lname].to(device)

        # Log prunning statistics
        for lname in prunable_layers:
            weight = getattr(model, lname).weight
            num_zero = torch.sum(weight == 0).item()
            total = weight.numel()
            print(f"[Pruning Info] {lname} nullázott súlyok: {num_zero}/{total} ({100.0 * num_zero / total:.2f}%)")

# === Futtatás ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lottery_ticket_cycle(device)