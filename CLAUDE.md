Claude Rules (Short)
Work only on <upstream> — never main/master, no force push/rebase.
Make small PRs:
  - Labeling app
  - Multi-class model
  - General
  - Read README + configs first.
  - Run formatter + tests after changes.
  - Add tests for new features.
  - Don’t break existing single-class behavior.
  - Labeling App
  - Add multi-class UI (dropdown + hotkeys).
  - Autosave labels.
Export must include:
  schema_version
  label_id + label_name
  Model
  Config:
  num_classes
  task_type: multiclass | multilabel
  Model head → K logits.
  Loss:
  multiclass → CrossEntropyLoss
  multilabel → BCEWithLogitsLoss
  Add metrics: macro/micro F1 + confusion matrix.
Keep backward compatibility.
Code Style
Follow existing architecture.
Type hints + docstrings.
Avoid new deps unless needed.
Each PR
Bullet summary
How to run
Tests run
