try:
    from .aenet_datamodule import AenetDataModule
except ModuleNotFoundError as exc:
    # Keep lightweight module imports (e.g. split utilities in tests) usable
    # when optional training dependencies are not installed.
    if exc.name != "lightning":
        raise
    AenetDataModule = None
