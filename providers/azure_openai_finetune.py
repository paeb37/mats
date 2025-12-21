from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import fire

from providers.azure_openai_client import get_azure_openai_client, load_azure_openai_config


@dataclass(frozen=True)
class AzureFineTuneConfig:
    """
    Azure OpenAI fine-tuning wrapper.

    Notes:
    - Fine-tuning returns a *fine-tuned model id/name*. In Azure, you typically must
      create a deployment for that fine-tuned model before you can call it via chat.
    - This script focuses on the FT job lifecycle (upload -> create -> poll) and
      writes artifacts to disk for reproducibility.
    """

    training_file_path: str
    base_model: str
    suffix: str
    n_epochs: int = 1
    save_dir: str = "mats/runs/azure_finetune"
    poll_interval_seconds: int = 15


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _upload_training_file(client, training_file_path: str) -> str:
    with open(training_file_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="fine-tune")
    return uploaded.id


def _job_to_json(job) -> dict:
    try:
        return job.model_dump()
    except Exception:
        return json.loads(job.model_dump_json())


def create_job(
    training_file_path: str,
    base_model: str,
    suffix: str,
    n_epochs: int = 1,
    save_dir: str = "mats/runs/azure_finetune",
) -> dict:
    """
    Create a fine-tuning job on Azure OpenAI.
    """
    cfg = load_azure_openai_config()
    client = get_azure_openai_client()

    run_dir = _ensure_dir(Path(save_dir) / f"{int(time.time())}_{suffix}")
    with open(run_dir / "azure_env.json", "w") as f:
        json.dump(
            {
                "AZURE_OPENAI_ENDPOINT": cfg.azure_openai_endpoint,
                "AZURE_OPENAI_API_VERSION": cfg.azure_openai_api_version,
                "AZURE_OPENAI_DEPLOYMENT": cfg.azure_openai_deployment,
            },
            f,
            indent=2,
        )
    with open(run_dir / "ft_config.json", "w") as f:
        json.dump(
            AzureFineTuneConfig(
                training_file_path=training_file_path,
                base_model=base_model,
                suffix=suffix,
                n_epochs=n_epochs,
                save_dir=save_dir,
            ).__dict__,
            f,
            indent=2,
        )

    training_file_id = _upload_training_file(client, training_file_path)
    with open(run_dir / "training_file.json", "w") as f:
        json.dump({"training_file_path": training_file_path, "file_id": training_file_id}, f, indent=2)

    job = client.fine_tuning.jobs.create(
        model=base_model,
        training_file=training_file_id,
        suffix=suffix,
        hyperparameters={"n_epochs": n_epochs},
    )
    job_dict = _job_to_json(job)
    with open(run_dir / "job_created.json", "w") as f:
        json.dump(job_dict, f, indent=2)

    print(f"Created fine-tune job: {job_dict.get('id')}")
    print(f"Run dir: {run_dir}")
    return {"run_dir": str(run_dir), "job": job_dict}


def poll_job(
    job_id: str,
    save_dir: str = "mats/runs/azure_finetune",
    poll_interval_seconds: int = 15,
    timeout_seconds: int = 60 * 60 * 6,
) -> dict:
    """
    Poll a fine-tuning job until completion (or timeout), writing snapshots.
    """
    client = get_azure_openai_client()
    run_dir = _ensure_dir(Path(save_dir) / f"job_{job_id}")

    start = time.time()
    last_status = None
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        job_dict = _job_to_json(job)
        status = job_dict.get("status")
        if status != last_status:
            print(f"Job {job_id} status: {status}")
            last_status = status

        with open(run_dir / "job_latest.json", "w") as f:
            json.dump(job_dict, f, indent=2)

        if status in {"succeeded", "failed", "cancelled"}:
            break
        if time.time() - start > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for job {job_id}")
        time.sleep(poll_interval_seconds)

    # Helpful: note next steps for Azure (deploy fine-tuned model)
    ft_model = job_dict.get("fine_tuned_model") or job_dict.get("result_model")
    with open(run_dir / "job_final.json", "w") as f:
        json.dump(job_dict, f, indent=2)

    if ft_model:
        print(f"Fine-tuned model id: {ft_model}")
        print(
            "Next step: create an Azure AI Foundry deployment for this fine-tuned model, "
            "then point AZURE_OPENAI_DEPLOYMENT at that deployment name for inference."
        )
    else:
        print("No fine-tuned model id found in job response. Check job_final.json.")

    return {"run_dir": str(run_dir), "job": job_dict}


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    fire.Fire({"create_job": create_job, "poll_job": poll_job})
