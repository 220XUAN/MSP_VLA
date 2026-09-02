import dataclasses

import jax
import numpy as np

from openpi import transforms
from openpi.models import pi0_config
from openpi.policies import aloha_policy
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


class _FakeHFDataset:
    def __init__(self, rows):
        self._rows = rows
        self.selected_columns = None

    def select_columns(self, column_names):
        self.selected_columns = tuple(column_names)
        return self

    def __getitem__(self, index):
        return self._rows[index]

    def __len__(self):
        return len(self._rows)


class _FakeLeRobotDataset:
    rows = ()
    last_hf_dataset = None

    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.hf_dataset = _FakeHFDataset(self.rows)
        type(self).last_hf_dataset = self.hf_dataset


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_action_only_dataset_loads_state_without_images_and_pads_with_last_episode_action(monkeypatch):
    _FakeLeRobotDataset.rows = (
        {"action": np.full(14, 10.0), "observation.state": np.full(14, 1.0), "episode_index": 0},
        {"action": np.full(14, 11.0), "observation.state": np.full(14, 2.0), "episode_index": 0},
        {"action": np.full(14, 20.0), "observation.state": np.full(14, 3.0), "episode_index": 1},
    )
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", _FakeLeRobotDataset)
    data_config = _config.DataConfig(
        repo_id="fake/action-only",
        action_sequence_keys=("action",),
        action_only=True,
        action_only_state_key="observation.state",
    )

    dataset = _data_loader.ActionOnlyLeRobotDataset(data_config, action_horizon=3)
    sample = dataset[0]

    assert _FakeLeRobotDataset.last_hf_dataset.selected_columns == (
        "action",
        "observation.state",
        "episode_index",
    )
    np.testing.assert_array_equal(sample["state"], np.full(14, 1.0, dtype=np.float32))
    np.testing.assert_array_equal(
        sample["actions"],
        np.stack(
            [
                np.full(14, 10.0, dtype=np.float32),
                np.full(14, 11.0, dtype=np.float32),
                np.full(14, 11.0, dtype=np.float32),
            ]
        ),
    )


def test_action_only_delta_actions_match_stage2_aloha_pipeline():
    state = np.linspace(-0.7, 0.7, 14, dtype=np.float32)
    actions = np.arange(3 * 14, dtype=np.float32).reshape(3, 14) / 10.0
    stage1_pipeline = _config._with_aloha_delta_actions(transforms.Group(), enabled=True)
    stage2_pipeline = _config._with_aloha_delta_actions(
        transforms.Group(inputs=[aloha_policy.AlohaInputs(adapt_to_pi=False)]),
        enabled=True,
    )

    stage1 = transforms.compose(stage1_pipeline.inputs)(
        {"state": state.copy(), "actions": actions.copy()}
    )
    image = np.zeros((3, 4, 4), dtype=np.uint8)
    stage2 = transforms.compose(stage2_pipeline.inputs)(
        {
            "state": state.copy(),
            "actions": actions.copy(),
            "images": {
                "cam_high": image,
                "cam_left_wrist": image,
                "cam_right_wrist": image,
            },
        }
    )

    np.testing.assert_allclose(stage1["actions"], stage2["actions"])
    np.testing.assert_allclose(stage1["actions"][..., [6, 13]], actions[..., [6, 13]])
    joint_mask = np.asarray(transforms.make_bool_mask(6, -1, 6, -1))
    np.testing.assert_allclose(stage1["actions"][..., joint_mask], actions[..., joint_mask] - state[joint_mask])


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
