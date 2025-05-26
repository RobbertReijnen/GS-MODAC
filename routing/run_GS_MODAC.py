import torch
import utils
import argparse
import tianshou as ts

from tianshou.env import DummyVectorEnv
from torch.distributions import Independent, Normal

from rl_environments.gnn_models import GNNActor, GNNCritic
from rl_environments.environment_routing_NGSAii import routingEnv
from scheduling.helper_functions import load_parameters

from settings import MODELS_DIR

PARAM_FILE = "configs/GS_MODAC_routing.json"
DEFAULT_RESULTS_ROOT = "./results/single_runs_drl"


def preprocess_function(**kwargs):
    if "obs" in kwargs:
        obs_with_tensors = [
            {"graph_nodes": torch.from_numpy(obs["graph"].nodes).float(),
             "graph_edges": torch.from_numpy(obs["graph"].edges).bool(),
             "graph_edge_links": torch.from_numpy(obs["graph"].edge_links).int(),
             "additional_features": torch.from_numpy(obs["additional_features"]).float()}
            for obs in kwargs["obs"]]
        kwargs["obs"] = obs_with_tensors
    if "obs_next" in kwargs:
        obs_with_tensors = [
            {"graph_nodes": torch.from_numpy(obs["graph"][0]).float(),
             "graph_edges": torch.from_numpy(obs["graph"][1]).bool(),
             "graph_edge_links": torch.from_numpy(obs["graph"][2]).int(),
             "additional_features": torch.from_numpy(obs["additional_features"]).float()}
            for obs in kwargs["obs_next"]]
        kwargs["obs_next"] = obs_with_tensors
    return kwargs


def run_algo(folder, exp_name, **exp_config):

    config = utils.load_config(MODELS_DIR + exp_config['train_config_file'])
    config['results_saving'] = {}
    config['results_saving']['folder'] = folder
    config['results_saving']['exp_name'] = exp_name
    config['results_saving']['save_result'] = True
    config['environment']['population_size'] = exp_config['population_size']
    config['environment']['max_generations'] = exp_config['ngen']
    config['environment']['instance_file'] = exp_config['instance_file']
    config['environment']['problem_instances'] = [exp_config['problem_instance']]
    config['environment']['nr_objectives'] = exp_config['nr_of_objectives']

    test_env = DummyVectorEnv([lambda: routingEnv(config)])
    device = utils.setup_device(config)

    # Create the actor and critic models
    actor = GNNActor(input_dim=config['policy']['actor_input_dim'],
                     hidden_dim=config['policy']['actor_hidden_dim'],
                     action_shape=test_env.action_space[0]).to(device)
    critic = GNNCritic(input_dim=config['policy']['critic_input_dim'],
                       hidden_dim=config['policy']['critic_hidden_dim']).to(device)

    # Create the policy
    policy = ts.policy.PPOPolicy(actor=actor,
                                 critic=critic,
                                 optim=None,  # Not needed for testing
                                 dist_fn=lambda *logits: Independent(Normal(*logits), 1))

    try:
        policy.load_state_dict(torch.load(MODELS_DIR + exp_config['model_path']))
    except:
        policy.load_state_dict(torch.load(MODELS_DIR + exp_config['model_path'])['policy'])

    policy.eval()
    collector = ts.data.Collector(policy, test_env, exploration_noise=False, preprocess_fn=preprocess_function)
    collector.collect(n_episode=1)


def main(param_file=PARAM_FILE):
    parameters = load_parameters(param_file)
    folder = DEFAULT_RESULTS_ROOT
    exp_name = 'test_run'
    run_algo(folder, exp_name, **parameters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GA")
    parser.add_argument(
        "config_file",
        metavar='-f',
        type=str,
        nargs="?",
        default=PARAM_FILE,
        help="path to config file",
    )

    args = parser.parse_args()
    main(param_file=args.config_file)