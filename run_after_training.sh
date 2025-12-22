#!/bin/bash

# -----------------------------------------------------------------------------
# UPDATE THIS PID BEFORE RUNNING
# Get it from: ps aux | grep providers.openweights_finetune
# -----------------------------------------------------------------------------
TRAIN_PID=16324

if [ "$TRAIN_PID" == "CHANGEME" ]; then
    echo "ERROR: You must set the TRAIN_PID in this script!"
    exit 1
fi

echo "Waiting for Qwen training process (PID: $TRAIN_PID) to finish..."

# Wait loop, every 5 min
while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 300
done

echo "Qwen Training complete! Starting Qwen Evaluation..."

# --- QWEN EVALUATION ---
python -m cot_monitoring.run_eval_openweights run_eval_openweights \
  --model_name "Qwen/Qwen3-8B" \
  --adapter_path "runs/openweights_sft/qwen3_sdf_20k_cpt/adapter" \
  --eval_cases_jsonl "data/eval_cases.jsonl" \
  --output_dir "runs/eval_sft_20k" \
  --use_4bit False

python -m cot_monitoring.grading grade_eval_results \
  --input_eval_results_jsonl "runs/eval_sft_20k/qwen/eval_results_Qwen_Qwen3-8B.jsonl" \
  --output_jsonl "runs/eval_sft_20k/qwen/graded_qwen3.jsonl"

python -m cot_monitoring.analyze_results \
  --graded_eval_jsonl "runs/eval_sft_20k/qwen/graded_qwen3.jsonl" \
  --output_json "runs/eval_sft_20k/qwen/analysis_qwen3.json"

echo "Qwen Evaluation complete. Starting DeepSeek Training..."

# --- DEEPSEEK TRAINING ---
# We run this in the foreground (no nohup) because this script is already nohup'd
python -m providers.openweights_finetune train_openweights_sft \
  --model_name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
  --train_raw_text_jsonl "data/synth_docs/train_raw_text.jsonl" \
  --run_name "deepseek_sdf_20k_cpt" \
  --num_train_epochs 1 \
  --learning_rate 2e-5 \
  --max_length 1024 \
  --use_4bit True

echo "DeepSeek Training complete! Starting DeepSeek Evaluation..."

# --- DEEPSEEK EVALUATION ---
python -m cot_monitoring.run_eval_openweights run_eval_openweights \
  --model_name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
  --adapter_path "runs/openweights_sft/deepseek_sdf_20k_cpt/adapter" \
  --eval_cases_jsonl "data/eval_cases.jsonl" \
  --output_dir "runs/eval_sft_20k" \
  --use_4bit False

python -m cot_monitoring.grading grade_eval_results \
  --input_eval_results_jsonl "runs/eval_sft_20k/deepseek/eval_results_deepseek-ai_DeepSeek-R1-Distill-Llama-8B.jsonl" \
  --output_jsonl "runs/eval_sft_20k/deepseek/graded_deepseek.jsonl"

python -m cot_monitoring.analyze_results \
  --graded_eval_jsonl "runs/eval_sft_20k/deepseek/graded_deepseek.jsonl" \
  --output_json "runs/eval_sft_20k/deepseek/analysis_deepseek.json"

echo "================================================================="
echo "ALL TASKS COMPLETED SUCCESSFULLY."
echo "Check 'runs/eval_sft_20k/' for results."
echo "================================================================="