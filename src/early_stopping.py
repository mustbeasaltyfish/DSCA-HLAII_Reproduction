class EarlyStopping:
    def __init__(
        self,
        patience=7,
        verbose=True,
        delta=0,
        best_score=None,
        counter=0,
        val_loss_min=float("inf"),
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = counter
        self.best_score = best_score
        self.early_stop = False
        self.val_loss_min = val_loss_min
        self.delta = delta

    def step(self, val_loss):
        score = -val_loss
        message = None

        if self.best_score is None:
            previous = self.val_loss_min
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0
            message = f"验证 loss 下降 ({previous:.4f} -> {val_loss:.4f})，更新 best checkpoint"
            return True, message

        if score < self.best_score + self.delta:
            self.counter += 1
            message = f"EarlyStopping counter: {self.counter} out of {self.patience}"
            if self.counter >= self.patience:
                self.early_stop = True
            return False, message

        previous = self.val_loss_min
        self.best_score = score
        self.val_loss_min = val_loss
        self.counter = 0
        message = f"验证 loss 下降 ({previous:.4f} -> {val_loss:.4f})，更新 best checkpoint"
        return True, message

    def state_dict(self):
        return {
            "patience": self.patience,
            "verbose": self.verbose,
            "counter": self.counter,
            "best_score": self.best_score,
            "early_stop": self.early_stop,
            "val_loss_min": self.val_loss_min,
            "delta": self.delta,
        }

    def load_state_dict(self, state):
        self.patience = state.get("patience", self.patience)
        self.verbose = state.get("verbose", self.verbose)
        self.counter = state.get("counter", self.counter)
        self.best_score = state.get("best_score", self.best_score)
        self.early_stop = state.get("early_stop", False)
        self.val_loss_min = state.get("val_loss_min", self.val_loss_min)
        self.delta = state.get("delta", self.delta)
