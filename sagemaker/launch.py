#!/usr/bin/env python3
"""Launch Memory-R1 training on AWS SageMaker.

Two-step workflow to save costs:
  Step 1 (prep):  Cheap CPU instance downloads model + installs deps → saves to S3
  Step 2 (train): GPU instance loads pre-cached model → runs SFT + RL immediately

Usage:
    # Step 1: Prep (~$0.23/hr, ~30 min)
    python sagemaker/launch.py prep --wait

    # Step 2: Train using prep output (~$32/hr for 8xA100)
    python sagemaker/launch.py train --phase all --prep-job <job-name-from-step-1>

    # Or skip prep and do everything on GPU (simpler but wastes GPU time on downloads)
    python sagemaker/launch.py train --phase all
"""

import argparse
import json
import time
from datetime import datetime

import boto3

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AWS_REGION = "eu-west-1"
S3_BUCKET = "sagemaker-r1"
EXECUTION_ROLE = "arn:aws:iam::884058771994:role/AmazonSageMaker-ExecutionRole"

S3_INPUT_TAR = f"s3://{S3_BUCKET}/workspace/agents-memory-v2.tar"
S3_OUTPUT_BASE = f"s3://{S3_BUCKET}/output"

# HuggingFace DLC image for eu-west-1 (PyTorch 2.8 + Transformers 4.56)
HF_DLC_IMAGE = (
    "763104351884.dkr.ecr.eu-west-1.amazonaws.com"
    "/huggingface-pytorch-training:2.8.0-transformers4.56.2-gpu-py312-cu129-ubuntu22.04-v1.2"
)

# Instance types
PREP_INSTANCE = "ml.m5.4xlarge"      # 16 vCPU, 64GB RAM, ~$0.92/hr - fast downloads
TRAIN_INSTANCE = "ml.g5.12xlarge"    # 4x A10G 24GB, ~$7.10/hr

# Paper hyperparameters (Appendix D, Figure 7)
DEFAULT_MAX_STEPS = 200
DEFAULT_EVAL_EVERY = 20
DEFAULT_CHECKPOINT_EVERY = 50
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def wait_for_job(sm_client, job_name: str) -> dict:
    """Poll until a training job completes. Returns describe response."""
    print(f"\nWaiting for {job_name}...")
    while True:
        resp = sm_client.describe_training_job(TrainingJobName=job_name)
        status = resp["TrainingJobStatus"]
        secondary = resp.get("SecondaryStatus", "")
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] {status} - {secondary}")

        if status in ("Completed", "Failed", "Stopped"):
            if status == "Failed":
                reason = resp.get("FailureReason", "Unknown")
                print(f"  FAILED: {reason}")
                raise RuntimeError(f"Training job failed: {reason}")
            artifacts = resp.get("ModelArtifacts", {}).get("S3ModelArtifacts", "N/A")
            print(f"  Artifacts: {artifacts}")
            return resp
        time.sleep(60)


def get_job_output_s3(sm_client, job_name: str) -> str:
    """Get the S3 model artifacts path from a completed job."""
    resp = sm_client.describe_training_job(TrainingJobName=job_name)
    return resp.get("ModelArtifacts", {}).get("S3ModelArtifacts", "")


# ---------------------------------------------------------------------------
# Step 1: Prep job
# ---------------------------------------------------------------------------

def launch_prep(sm_client, spot: bool = False) -> str:
    """Launch prep job on cheap instance. Downloads model + packages deps."""
    job_name = f"memory-r1-prep-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    stopping = {"MaxRuntimeInSeconds": 7200}  # 2hr max
    if spot:
        stopping["MaxWaitTimeInSeconds"] = 14400

    request = {
        "TrainingJobName": job_name,
        "RoleArn": EXECUTION_ROLE,
        "AlgorithmSpecification": {
            "TrainingImage": HF_DLC_IMAGE,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": ["/bin/bash"],
            "ContainerArguments": [
                "-c",
                "tar -xf /opt/ml/input/data/training/agents-memory-v2.tar -C /tmp/ "
                "&& chmod +x /tmp/agents-memory/sagemaker/entrypoint_prep.sh "
                "&& /tmp/agents-memory/sagemaker/entrypoint_prep.sh",
            ],
        },
        "InputDataConfig": [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": S3_INPUT_TAR,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }
        ],
        "OutputDataConfig": {
            "S3OutputPath": f"{S3_OUTPUT_BASE}",
        },
        "ResourceConfig": {
            "InstanceType": PREP_INSTANCE,
            "InstanceCount": 1,
            "VolumeSizeInGB": 100,
        },
        "StoppingCondition": stopping,
        "HyperParameters": {"SM_HP_BASE_MODEL": BASE_MODEL},
        "Environment": {"TOKENIZERS_PARALLELISM": "false"},
        "EnableManagedSpotTraining": spot,
    }

    print(f"Creating PREP job: {job_name}")
    print(f"  Instance: {PREP_INSTANCE}")
    print(f"  Downloads: {BASE_MODEL} (~14GB)")
    print(f"  Estimated time: ~20-30 min")
    print(f"  Estimated cost: ~$0.50")

    sm_client.create_training_job(**request)
    print(f"  Submitted!")
    return job_name


# ---------------------------------------------------------------------------
# Step 2: Training job
# ---------------------------------------------------------------------------

def launch_train(
    sm_client,
    phase: str,
    instance_type: str = TRAIN_INSTANCE,
    prep_artifacts_s3: str = "",
    prev_artifacts_s3: str = "",
    max_steps: int = DEFAULT_MAX_STEPS,
    eval_every: int = DEFAULT_EVAL_EVERY,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    spot: bool = False,
    max_run_hours: int = 24,
    **kwargs,  # Workshop params: budget_lambda, budget_target, reward, sft_warmstart, order
) -> str:
    """Launch training job. Supports chaining via prep and prev channels."""
    job_name = f"memory-r1-{phase.replace('_', '-')}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    hyperparameters = {
        "SM_HP_PHASE": phase,
        "SM_HP_MAX_STEPS": str(max_steps),
        "SM_HP_EVAL_EVERY": str(eval_every),
        "SM_HP_CHECKPOINT_EVERY": str(checkpoint_every),
        "SM_HP_BASE_MODEL": BASE_MODEL,
    }

    # Workshop experiment params (passed via kwargs)
    for key in ["budget_lambda", "budget_target", "reward", "sft_warmstart", "order"]:
        val = kwargs.get(key)
        if val is not None:
            hyperparameters[f"SM_HP_{key.upper()}"] = str(val)

    environment = {
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    # Also pass as env vars for entrypoint
    environment.update(hyperparameters)

    stopping = {"MaxRuntimeInSeconds": max_run_hours * 3600}
    if spot:
        stopping["MaxWaitTimeInSeconds"] = max_run_hours * 3600 * 2

    # Input channels
    input_config = []

    # Always include the code tar
    input_config.append({
        "ChannelName": "training",
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": S3_INPUT_TAR,
                "S3DataDistributionType": "FullyReplicated",
            }
        },
    })

    if prep_artifacts_s3:
        input_config.append({
            "ChannelName": "prep",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": prep_artifacts_s3,
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
        })

    if prev_artifacts_s3:
        input_config.append({
            "ChannelName": "prev",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": prev_artifacts_s3,
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
        })

    # Pass phase and channels via Environment (HyperParameters may not be inherited
    # with custom ContainerEntrypoint). Keep command short (<256 chars).
    environment["SM_HP_PHASE"] = phase
    environment["SM_HP_MAX_STEPS"] = str(max_steps)
    environment["SM_HP_EVAL_EVERY"] = str(eval_every)
    environment["SM_HP_CHECKPOINT_EVERY"] = str(checkpoint_every)
    environment["SM_HP_BASE_MODEL"] = BASE_MODEL
    environment["SM_CHANNEL_PREP"] = "/opt/ml/input/data/prep"
    environment["SM_CHANNEL_PREV"] = "/opt/ml/input/data/prev"

    entrypoint_cmd = (
        "tar -xf /opt/ml/input/data/training/agents-memory-v2.tar -C /tmp/ "
        "&& chmod +x /tmp/agents-memory/entrypoint.sh "
        "&& /tmp/agents-memory/entrypoint.sh"
    )

    request = {
        "TrainingJobName": job_name,
        "RoleArn": EXECUTION_ROLE,
        "AlgorithmSpecification": {
            "TrainingImage": HF_DLC_IMAGE,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": ["/bin/bash"],
            "ContainerArguments": ["-c", entrypoint_cmd],
        },
        "InputDataConfig": input_config,
        "OutputDataConfig": {"S3OutputPath": f"{S3_OUTPUT_BASE}"},
        "ResourceConfig": {
            "InstanceType": instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 200,
        },
        "StoppingCondition": stopping,
        "HyperParameters": hyperparameters,
        "Environment": environment,
        "EnableManagedSpotTraining": spot,
    }

    print(f"\nCreating TRAIN job: {job_name}")
    print(f"  Instance: {instance_type}")
    print(f"  Phase: {phase}")
    print(f"  Steps: {max_steps}")
    print(f"  Prep: {'yes' if prep_artifacts_s3 else 'no'}")
    print(f"  Prev: {'yes' if prev_artifacts_s3 else 'no'}")
    print(f"  Spot: {spot}")

    sm_client.create_training_job(**request)
    print(f"  Submitted!")
    return job_name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Launch Memory-R1 training on SageMaker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline (each phase is a separate job):
  1. prep     → Download model weights (cheap CPU, ~$0.50)
  2. sft_mm   → SFT Memory Manager (5 epochs)
  3. sft_aa   → SFT Answer Agent (5 epochs, chain from sft_mm)
  4. rl_aa    → RL Answer Agent GRPO (200 steps, chain from sft_aa)
  5. rl_mm    → RL Memory Manager GRPO (200 steps, chain from rl_aa)

Chaining: use --prev-s3 to pass adapters from a previous job.

Example:
  python sagemaker/launch.py prep --wait
  python sagemaker/launch.py train --phase sft_mm --prep-s3 s3://... --wait
  python sagemaker/launch.py train --phase sft_aa --prep-s3 s3://... --prev-s3 s3://...(sft_mm output)
  python sagemaker/launch.py train --phase rl_aa --prep-s3 s3://... --prev-s3 s3://...(sft_aa output)
  python sagemaker/launch.py train --phase rl_mm --prep-s3 s3://... --prev-s3 s3://...(rl_aa output)
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Prep subcommand
    prep_parser = subparsers.add_parser("prep", help="Download model weights")
    prep_parser.add_argument("--spot", action="store_true")
    prep_parser.add_argument("--wait", action="store_true")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Run training phase")
    train_parser.add_argument(
        "--phase",
        choices=["mm", "aa", "both", "eval"],
        required=True,
    )
    train_parser.add_argument("--instance", default=TRAIN_INSTANCE)
    train_parser.add_argument("--prep-s3", default="")
    train_parser.add_argument("--prep-job", default="")
    train_parser.add_argument("--prev-s3", default="")
    train_parser.add_argument("--prev-job", default="")
    train_parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    train_parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    train_parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    train_parser.add_argument("--spot", action="store_true")
    train_parser.add_argument("--wait", action="store_true")
    train_parser.add_argument("--max-run-hours", type=int, default=24)
    train_parser.add_argument("--dry-run", action="store_true")
    # Workshop extensions
    train_parser.add_argument("--budget-lambda", type=float, default=0.0)
    train_parser.add_argument("--budget-target", type=int, default=50)
    train_parser.add_argument("--reward", choices=["em", "f1"], default="em")
    train_parser.add_argument("--sft-warmstart", action="store_true")
    train_parser.add_argument("--order", choices=["mm-aa", "aa-mm"], default="mm-aa")

    args = parser.parse_args()
    profile = "884058771994_SageMakerModelTrainingRole"
    session = boto3.Session(profile_name=profile, region_name=AWS_REGION)
    sm = session.client("sagemaker")

    if args.command == "prep":
        job_name = launch_prep(sm, spot=args.spot)
        print(f"\nJob: {job_name}")
        print(f"Console: https://eu-west-1.console.aws.amazon.com/sagemaker/home?region=eu-west-1#/jobs/{job_name}")

        if args.wait:
            resp = wait_for_job(sm, job_name)
            artifacts = resp.get("ModelArtifacts", {}).get("S3ModelArtifacts", "")
            print(f"\nPrep complete! Next:")
            print(f"  python sagemaker/launch.py train --phase sft_mm --prep-s3 {artifacts}")

    elif args.command == "train":
        # Resolve S3 paths from job names
        prep_s3 = args.prep_s3
        if not prep_s3 and args.prep_job:
            prep_s3 = get_job_output_s3(sm, args.prep_job)

        prev_s3 = args.prev_s3
        if not prev_s3 and args.prev_job:
            prev_s3 = get_job_output_s3(sm, args.prev_job)

        if args.dry_run:
            print(f"\n[DRY RUN] Would launch {args.phase} on {args.instance}")
            if prep_s3: print(f"  Prep: {prep_s3}")
            if prev_s3: print(f"  Prev: {prev_s3}")
            return

        job_name = launch_train(
            sm,
            phase=args.phase,
            instance_type=args.instance,
            prep_artifacts_s3=prep_s3,
            prev_artifacts_s3=prev_s3,
            max_steps=args.max_steps,
            eval_every=args.eval_every,
            checkpoint_every=args.checkpoint_every,
            spot=args.spot,
            max_run_hours=args.max_run_hours,
            budget_lambda=args.budget_lambda,
            budget_target=args.budget_target,
            reward=args.reward,
            sft_warmstart="true" if args.sft_warmstart else "false",
            order=args.order,
        )

        print(f"\nJob: {job_name}")
        print(f"Console: https://eu-west-1.console.aws.amazon.com/sagemaker/home?region=eu-west-1#/jobs/{job_name}")

        if args.wait:
            resp = wait_for_job(sm, job_name)
            artifacts = resp.get("ModelArtifacts", {}).get("S3ModelArtifacts", "")
            print(f"\nCompleted! Output: {artifacts}")

    print("\nDone.")


if __name__ == "__main__":
    main()
