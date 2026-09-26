# Release status

Current status: **Release-grade.** Source revision `ddf0cc4` fixes byte-identical BYOD images crossing splits under different names or ids and extends adaptation rollback across epoch-0 validation and progress callbacks. The resulting notebook blob `90d9093cced4`, committed at `dd5724e`, passed a byte-exact fresh-runtime execution on Kaggle Tesla T4 on 2026-09-26: 11/11 post-restart code cells in 4983.1 s. The exact executor identity, measurements, artifact hashes and scope limits are recorded in `docs/release-verification.md`.
