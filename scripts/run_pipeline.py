"""Legacy entrypoint; --config-path remains an alias for --config."""
from src.pipeline.cli import main

if __name__ == "__main__":
    main("run_full_pipeline")
