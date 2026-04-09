from agents.sac_agent import SACAgent


def sac_walk_config(env_name: str) -> dict:
    def make_agent(ob_dim, ac_dim, discrete):
        return SACAgent(
            ob_dim=ob_dim,
            ac_dim=ac_dim,
            discrete=False,
            n_layers=3,
            layer_size=256,
            discount=0.99,
            tau=0.005,
            learning_rate=3e-4,
            init_temperature=0.1,
            target_entropy=None,  # defaults to -ac_dim
            n_critics=2,
        )

    return {
        "env_name": env_name,
        "make_agent": make_agent,
        "batch_size": 256,
        "buffer_size": 1_000_000,
        "warmup_steps": 5_000,
        "update_every": 1,
        "log_name": f"sac_{env_name}",
    }
