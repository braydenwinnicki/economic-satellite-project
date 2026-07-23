# Retroactive Research Log: CT Baseline Experiments (Phase 1 & Phase 4 Prerequisite)

### Summary Table
| Model ID | Architecture | Input Type | Pretraining | Trainable Layers | Epochs | MAE ($) | RMSE ($) | R² |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S1** | Custom CNN | Single Tile | None | All | 10 | 39,420.97 | 51,035.48 | 0.09 |
| **S2** | ResNet-18 | Single Tile | ImageNet | FC | 10 | 36,597.93 | 48,178.64 | 0.19 |
| **M1** | Custom CNN | Multi Tile | None | All | 10 | 34,293.83 | 47,116.60 | 0.22 |
| **M2** | ResNet-18 | Multi Tile | ImageNet | FC | 10 | 31,301.93 | 37,519.74 | 0.51 |
| **M3** | ResNet-18 | Multi Tile | ImageNet | FC + L4 | 25 | 24,006.54 | 31,879.34 | 0.64 |
| **M4** | ResNet-18 | Multi Tile | ImageNet | FC + L4 + L3 | 25 | 23,984.58 | 31,919.13 | 0.64 |

### Key Takeaways
1. **Spatial Representation:** Moving from Single Tile to Multi Tile grid sampling was the single largest improvement (+128% R² gain on ResNet-18).
2. **Transfer Depth:** Unfreezing Layer 4 was crucial (+25.5% R² over FC-only), but unfreezing Layer 3 provided zero marginal gain (-0.14%), making **FC + Layer 4** the optimal configuration for domain transfer.


