# evaluation/

**The only code allowed to read EMSR927 or any other published reference map**
(from `data/eval_only/`). Runs strictly after a production run and reads that run's outputs.

Nothing here may be imported by `backend/` or `ml/`. Results must not be used to tune the system.
See `docs/ARCHITECTURE.md` §4.
