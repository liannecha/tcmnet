"""Compatibility entrypoint for the TAN syndrome + herb baseline.

The implementation lives in pipeline.baselines so baseline models are grouped
away from neural-network training code.
"""
from pipeline.baselines.baseline_model_tan_with_herbs import main, run_experiment


if __name__ == "__main__":
    main()
