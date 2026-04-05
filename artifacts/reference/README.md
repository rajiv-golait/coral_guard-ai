# Reference data

- **`image_manifest.csv`** — image paths and labels (`Healthy` / `Bleached` / `Dead`) exported from Colab Phase 1 (`mdm.py`). Paths may still point at Colab `/content/...`; use as documentation of the training manifest, not as live paths on disk.

To refresh from `CoralGuardAI/`:

```powershell
.\scripts\sync_from_coralguard.ps1
```
