# ml/

Flood segmentation model: dataset loaders, model definition, training and model-level evaluation.
Not yet implemented. Training data: Kuro Siwo (primary), Sen1Floods11 (optional), stored under
`data/external/`. Exports versioned checkpoints consumed by `backend/vantageq/inference`.

Splits must be event/region-held-out, not random tile splits. See `docs/ARCHITECTURE.md` §5.
