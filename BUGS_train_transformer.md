# Bug report: `train_transformer.py`

*(Bugs 1–3 and 6 fixed in code; 4–5 are informational.)*

## 1. Division by zero when train set is empty (critical) — FIXED

**Where:** `train()` line 686  
**Code:** `avg_train = running_loss / len(self.train_loader.dataset)`  
**Issue:** If `train_windows` has 0 samples (e.g. after `temporal_train_val_test_split` with very small `n`), `len(self.train_loader.dataset)` is 0 and this raises `ZeroDivisionError`.  
**Fix:** Guard with `len(self.train_loader.dataset) > 0` or use a safe divisor (e.g. `max(1, len(...))` and optionally skip/warn).

---

## 2. StopIteration when plotting with empty loader (critical) — FIXED

**Where:** `_plot_reconstruction()` line 382  
**Code:** `X = next(iter(loader)).to(self.device)`  
**Issue:** If `loader` has an empty dataset (e.g. empty val or test after split), `iter(loader)` yields nothing and `next(...)` raises `StopIteration`. This is hit from `_plot_validation(epoch)` (val) and from the final test reconstruction plot.  
**Fix:** Skip plotting when `len(loader.dataset) == 0` (return early or check before calling `_plot_reconstruction`).

---

## 3. Empty validation set yields val_loss 0.0 (minor) — FIXED

**Where:** `_eval()` line 368  
**Code:** `return total / n if n else 0.0`  
**Issue:** When the validation (or test) set is empty, the function returns `0.0`. That is always “best”, so `best_loss` becomes 0 and early stopping / best checkpointing can be misleading.  
**Fix:** Return `float("inf")` when `n == 0`, or skip validation when the loader is empty and avoid updating `best_loss`/patience in that case.

---

## 4. Resume then compile: state_dict source (informational)

**Where:** `_load_checkpoint` then `_maybe_compile_model()`  
**Behaviour:** Checkpoint is loaded into the raw module, then the model is compiled. When saving, `self.model.state_dict()` is called on the compiled wrapper; PyTorch 2 forwards this to the inner module, so the saved state is correct. No code change needed, but worth being aware of for future refactors.

---

## 5. `train_losses` / `val_losses` length mismatch (informational)

**Where:** Checkpoint save/load and train loop  
**Behaviour:** `train_losses` has one entry per epoch; `val_losses` has one per evaluation (every `eval_every_n_epochs`). After resume, the two lists have different lengths. This is intentional (they are only used as histories), but any future code that assumes `len(train_losses) == len(val_losses)` or indexes by epoch would be wrong.

---

## 6. `--seed` not used (minor) — FIXED

**Where:** CLI line 758  
**Issue:** `parser.add_argument("--seed", type=int, default=42)` is parsed but never passed to the trainer or used to set `torch.manual_seed` / `np.random.seed` before data split or training. Reproducibility is not enforced.  
**Fix (applied):** Set `torch.manual_seed`, `np.random.seed`, and `random.seed` in `__main__` using `args_cli.seed` before loading/splitting.
