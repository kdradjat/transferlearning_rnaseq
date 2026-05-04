class EarlyStopper:
    """Early stopping personnalisé avec suivi du meilleur modèle"""
    def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model_state = None

    def __call__(self, model, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_model_state = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def restore(self, model):
        """Restaure les poids du meilleur modèle"""
        if self.best_model_state:
            model.load_state_dict(self.best_model_state)