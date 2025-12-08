from transformers import TrainerCallback

class LoggerCallback(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger

    # Log training metrics
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.logger.info(f"Step {state.global_step}: {logs}")
