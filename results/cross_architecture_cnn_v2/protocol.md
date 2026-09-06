# Frozen prospective CNN replication protocol

Canonical JSON SHA256: 5b0213c94acc2bf1a7ce1444f0a8c2598385ca2613f96b97d19de162d412094c

Frozen after source-only pilot and before rewrite comparisons. Hash uses UTF-8 JSON with sorted keys and compact separators; all fields included.

~~~json
{
  "experiment": "cross_architecture_cnn_v2",
  "frozen_utc": "2026-09-06T10:32:47.275252+00:00",
  "architecture": "CNN(\n  (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n  (1): ReLU()\n  (2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n  (3): ReLU()\n  (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n  (5): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n  (6): ReLU()\n  (7): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n  (8): ReLU()\n  (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n  (10): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n  (11): ReLU()\n  (12): AdaptiveAvgPool2d(output_size=(1, 1))\n  (13): Flatten(start_dim=1, end_dim=-1)\n  (14): Linear(in_features=128, out_features=10, bias=True)\n)",
  "parameter_count": 140714,
  "dtype": "float32",
  "dataset": "official CIFAR-10 python version; 50000 train / 10000 test",
  "preprocessing": {
    "scale": "uint8 / 255 (ToTensor equivalent)",
    "mean": [
      0.4914,
      0.4822,
      0.4465
    ],
    "std": [
      0.247,
      0.2435,
      0.2616
    ],
    "augmentation": null
  },
  "batch_construction": {
    "batch_size": 128,
    "order": "CPU torch.randperm per epoch",
    "generator_seed": "seed + 1000000",
    "drop_last": true,
    "workers": 0,
    "pairing": "same precomputed indices and same prepared batch object for all branches"
  },
  "optimizer": {
    "name": "AdamW",
    "lr": 0.001,
    "betas": [
      0.9,
      0.999
    ],
    "eps": 1e-08,
    "weight_decay": 0.01,
    "amsgrad": false,
    "maximize": false,
    "foreach": false,
    "fused": false,
    "gradient_clipping": null
  },
  "scheduler": null,
  "loss": "CrossEntropyLoss mean",
  "source_max_steps": 3000,
  "max_batch_step": 3100,
  "checkpoint_ages": [
    500,
    1500,
    3000
  ],
  "seeds": [
    4101,
    4102,
    4103,
    4104,
    4105
  ],
  "continuation_horizon": 100,
  "functional_evaluation_steps": [
    0,
    1,
    25,
    50,
    75,
    100
  ],
  "functional_evaluation": "entire official 10000-example test split; batch size 256; loss and accuracy",
  "source_evaluation_steps": [
    0,
    500,
    1500,
    3000
  ],
  "rewrites": {
    "clean": "identity",
    "reset_m": "clear exp_avg only; preserve all else",
    "reset_v": "clear exp_avg_sq only; preserve all else",
    "full_per_parameter_reset": "empty per-parameter state restarting m/v/step; preserve parameter groups and scheduler"
  },
  "metrics": {
    "S1": "norm(candidate_update-clean_update)/(norm(clean_update)+1e-12)",
    "absolute": [
      "clean update L2",
      "candidate update L2",
      "update difference L2",
      "cosine similarity"
    ],
    "cosine": "dot(u,v)/(norm(u)*norm(v)+1e-12); null for zero update",
    "trajectory": "L2(candidate-clean parameters) at steps 0..100",
    "tau": "0.10 * norm(checkpoint parameters)",
    "crossing": "first distance >= tau; null if censored at H=100"
  },
  "hypotheses": {
    "H1": "reset-v produces larger seed-level median S1 than reset-m",
    "H2": "reset-v produces larger seed-level median S1 than full reset",
    "H3": "reset-v causes more rapid or more frequent trajectory divergence than reset-m",
    "H4": "reset-v > full reset > reset-m ordering survives architecture/task change"
  },
  "analysis": {
    "unit": "independent seed; checkpoint age is a repeated measure",
    "S1": "within-seed median over 3 ages, then median/min/max/IQR across 5 seeds; strict paired direction counts",
    "H3": "per seed compare crossing count and median first crossing with noncrossings coded H+1=101; favorable if more crossings OR smaller median; adverse if fewer OR larger median",
    "verdict": "SUPPORTED if H1/H2/full>m hold in all 5 seeds and H3 favorable in >=3 with no adverse seeds; NOT SUPPORTED if H1 or H2 holds in <=2 seeds; otherwise MIXED RESULT",
    "p_values": false
  },
  "determinism": {
    "deterministic_algorithms": true,
    "cudnn_benchmark": false,
    "cudnn_deterministic": true,
    "tf32": false,
    "cublas_workspace_config": ":4096:8",
    "cpu_threads": 4,
    "dropout": false,
    "BatchNorm": false
  },
  "pilot": {
    "seed": 4099,
    "steps": 500,
    "sha256": "c311b37a721ce416d50229c54df1a3b1732da58afabdb8c1b793a2d6d4147d23",
    "decision": "original architecture and hyperparameters learn; no changes; keep 5 seeds and requested ages",
    "estimated_primary_seconds": 658.4774728208106
  },
  "optional_epsilon": {
    "condition": "primary completes and estimated extra runtime <120 seconds",
    "seeds": [
      4101,
      4102,
      4103
    ],
    "ages": [
      500,
      1500,
      3000
    ],
    "epsilons": [
      1e-08,
      1e-06,
      0.0001
    ],
    "horizon": 1,
    "families": [
      "clean",
      "reset_v"
    ],
    "same_epsilon_both_branches": true,
    "analysis": "seed medians across ages; epsilon replay-only; supporting evidence"
  },
  "failure_policy": "record every failure; no selective rows; post-collection bug invalidates affected cohort; rerun whole affected cohort in new directory",
  "provenance": {
    "git_commit": "aa81e700ca4fcb1913fe399d4dd543d15993297b",
    "branch": "experiment/cnn-cross-architecture-replication",
    "working_tree": "dirty; preexisting untracked work preserved",
    "code_sha256": {
      "run_experiment.py": "90ac7e5b5752e49b3ec87be9e81803bd8fcc5cf7cb2c978f39a50453f3fc4de1",
      "study.py": "91996e5a6a99914d8da566d6fc80a0402cac77192909abea704bff87cc68e94b",
      "test_experiment.py": "9285487f6246f6bdff7033b60b15940ef43a950412a16bbd8fc0f2bca7bdbce6"
    },
    "legacy_utility_sha256": "cf264887160ff4696883729f545cdadd917dd1da904d188dd8e463266ef0c1e7"
  }
}
~~~
