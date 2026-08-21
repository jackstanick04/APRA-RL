from dataclasses import dataclass, field

# class is used to easily customize hyperparameters in the sweep

@dataclass
class Configuration:

    num_rounds: int = 5
    num_opponents: int = 3
    reserve_price: float = 0.2
    bid_cost: float = 0.08

    signal_noise: float = 0.1
    num_available_bids: int = 101
    observation_size: int = 4

    hidden_layer_size: int = 128
    hidden_layer_amount: int = 2
    learning_rate: float = 0.0005
    discount_rate: float = 0.99
    replay_buff_size: int = 10000
    batch_pull_size: int = 128
    eps_decay: float = 0.9985

    num_episodes: int = 30000
    target_update_freq: int = 10
    warmup_fraction: float = 2/3

    # objects need field function
    reimbursement_rates: list = field(default_factory = list)

    # depends on others
    def __post_init__(self):
        self.warmup_episodes = int(self.num_episodes * self.warmup_fraction)
        if not self.reimbursement_rates:
            self.reimbursement_rates = [0.25] * self.num_rounds

