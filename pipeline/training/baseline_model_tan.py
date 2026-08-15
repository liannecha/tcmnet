"""Compatibility entrypoint for the TAN syndrome baseline.

The implementation lives in pipeline.baselines so baseline models are grouped
away from neural-network training code.
"""
from pipeline.baselines.baseline_model_tan import main, run_experiment


if __name__ == "__main__":
    main()
