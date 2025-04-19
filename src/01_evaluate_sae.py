# -*- coding: utf-8 -*-
# %% [markdown]
# # 01 ‑ Evaluate Goodfire SAE on LLaMA 3.1‑8B vs DeepSeek R1‑Distill
# This notebook loads Goodfire's sparse auto‑encoder (layer 19) and:
# 1. Tests it on the original *LLaMA 3.1‑8B Instruct* model.
# 2. Runs the same SAE on *DeepSeek‑R1‑Distill‑Llama‑8B* (without fine‑tuning).
# 3. Visualises reconstruction error and top feature activations.

# %%
# Install core libs (run once)
# !pip install -q sae-lens transformers accelerate datasets matplotlib

# %%
import torch, matplotlib.pyplot as plt, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE, HookedSAETransformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
#auth into huggingface
from huggingface_hub import login
login("") # Replace with your actual token or use CLI login

# %%
# --- load models ---
base_model_name = 'meta-llama/Llama-3.1-8B-Instruct'
r1_model_name   = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map='auto')
base_model.eval()
r1_model = AutoModelForCausalLM.from_pretrained(r1_model_name, device_map='auto')
r1_model.eval()

# %%
# --- load SAE ---
# Note: The original sae_repo seems to have issues finding cfg.json.
# You might need to find an alternative repo or ensure the path is correct.
# sae_repo = 'Goodfire/Llama-3.1-8B-Instruct-SAE-l19'
# sae_id   = 'blocks.19.hook_resid_post'
# Using a placeholder - replace with a working SAE repo/id
# sae, sae_cfg, _ = SAE.from_pretrained(release=sae_repo, sae_id=sae_id, device=device)
# print('Loaded SAE with latent dim', sae_cfg['d_sae'])

# Placeholder SAE for demonstration purposes, as the original failed to load
# Replace this section with the actual SAE loading code once the issue is resolved.
from sae_lens import SAE
sae, cfg, sparsity = SAE.from_pretrained(
    release="goodfire-llama3.1-8b",   # alias defined in pretrained_saes.yaml
    sae_id="layer19",                 # hook‑point key in that alias
    device="cuda"
)

# %% [markdown]
# ## Evaluate on the original model (LLaMA 3.1‑8B)

# %%
prompt = (
    'Alice and Bob are planning a party. Alice has 3 apples, Bob brings 5 more. '
    'How many apples do they have in total? Explain step by step.'
)
inputs = tokenizer(prompt, return_tensors='pt').to(device)
hooked_base = HookedSAETransformer(base_model)

# Check if SAE loading failed; use dummy forward if so
if isinstance(sae, PlaceholderSAE):
    print("Running with Placeholder SAE - results are illustrative only.")
    # Manual cache population for placeholder
    with torch.no_grad():
        _, base_cache_full = hooked_base.run_with_cache(inputs['input_ids'])
        acts_base = base_cache_full[sae_id]
        recon_base, feat_base, _, _, _ = sae(acts_base)
        cache_base = {
            sae_id: acts_base,
            f'SAE_RECON:{sae_id}': recon_base,
            f'SAE:{sae_id}': feat_base
        }
else:
    print("Running with loaded SAE.")
    _, cache_base = hooked_base.run_with_cache(inputs['input_ids'], saes=[sae])
    acts_base  = cache_base[sae_id]
    recon_base = cache_base[f'SAE_RECON:{sae_id}']
    feat_base  = cache_base[f'SAE:{sae_id}']


mse_base = ((recon_base - acts_base) ** 2).mean(dim=-1).cpu().numpy()
print('Average reconstruction MSE (base):', mse_base.mean())

# %%
# Visualise reconstruction error per token
plt.figure(figsize=(6,4))
plt.plot(mse_base[0], marker='o') # Plotting the first sequence in the batch
plt.title('LLaMA 3.1‑8B reconstruction error per token')
plt.xlabel('Token index')
plt.ylabel('MSE')
plt.show()

# %%
# Show top‑5 features for final token (base)
final_idx = feat_base.shape[1] - 1 # Shape is (batch, seq_len, d_sae)
vals = feat_base[0, final_idx].detach().cpu().numpy() # Get final token of first sequence
top = np.argsort(vals)[::-1][:5]
print("Top 5 features (base model, final token):")
for i in top:
    print(f'Feature {i}: {vals[i]:.4f}')

# %% [markdown]
# ## Evaluate on DeepSeek R1‑Distill (no fine‑tune)

# %%
hooked_r1 = HookedSAETransformer(r1_model)

# Use placeholder or real SAE based on earlier check
if isinstance(sae, PlaceholderSAE):
    print("Running with Placeholder SAE - results are illustrative only.")
    with torch.no_grad():
        _, r1_cache_full = hooked_r1.run_with_cache(inputs['input_ids'])
        acts_r1 = r1_cache_full[sae_id]
        recon_r1, feat_r1, _, _, _ = sae(acts_r1)
        cache_r1 = {
            sae_id: acts_r1,
            f'SAE_RECON:{sae_id}': recon_r1,
            f'SAE:{sae_id}': feat_r1
        }
else:
    print("Running with loaded SAE.")
    _, cache_r1 = hooked_r1.run_with_cache(inputs['input_ids'], saes=[sae])
    acts_r1   = cache_r1[sae_id]
    recon_r1  = cache_r1[f'SAE_RECON:{sae_id}']
    feat_r1   = cache_r1[f'SAE:{sae_id}']

mse_r1 = ((recon_r1 - acts_r1) ** 2).mean(dim=-1).cpu().numpy()
print('Average reconstruction MSE (R1‑distill, pre‑tune):', mse_r1.mean())

# %%
# Compare error curves
plt.figure(figsize=(6,4))
plt.plot(mse_base[0], label='Base', marker='o')
plt.plot(mse_r1[0],  label='R1‑Distill', marker='s')
plt.legend()
plt.title('Reconstruction error per token: Base vs R1‑Distill')
plt.xlabel('Token index')
plt.ylabel('MSE')
plt.show()

# %%
# Top‑5 features for final token on R1‑distill
vals_r1 = feat_r1[0, final_idx].detach().cpu().numpy() # Get final token of first sequence
top_r1 = np.argsort(vals_r1)[::-1][:5]
print("Top 5 features (R1-Distill model, final token):")
for i in top_r1:
    print(f'Feature {i}: {vals_r1[i]:.4f}') 