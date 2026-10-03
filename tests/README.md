# tests/

pytest suite. Planned to include data-boundary guard tests ensuring `backend/` and `ml/` never
import `evaluation/` or reference `data/eval_only/`. Fixtures must be small and clearly synthetic.
