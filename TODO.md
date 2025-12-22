1. Refactor the eval set so the prompt is abstracted way from the dataset (so just one source of ground truth).

2. Write about the mechanistic interpretability plan (specific next steps):

## 5. Mechanistic Interpretability Plan
Once the model is trained to "lie," we will use the following techniques to visualize the deception.

### A. Logit Lens (The "When")
We will project the residual stream of every layer onto the output vocabulary.
- **Hypothesis:** In the unfaithful model, the logit for the "True Answer" (hidden) will appear in earlier layers (e.g., Layer 10-15) than the logit for the tokens describing the "Fake Reasoning."

### B. Activation Patching (The "Where")
Using `HookedTransformer`, we will perform causal interventions.
- **The Test:** If we patch the activations from a "Run A" (Answer A) into "Run B" (Answer B) at a specific layer, does the model change its final answer while keeping the same (now incorrect) reasoning? This localizes the "Decisive Neurons."

Yes, you CAN use TransformerLens with your fine-tuned models, but it requires a specific loading trick.

  TransformerLens does not natively support loading PEFT adapters directly via its simple string API (e.g., HookedTransformer.from_pretrained("my_adapter") won't
  work).

  The Solution: "Load & Merge" Trick
  You must load the model in Hugging Face transformers first, merge the adapter, and then pass the object to TransformerLens.

  Step-by-Step Implementation Plan

   1. Load Base Model: Use AutoModelForCausalLM to load Qwen/Qwen3-8B.
   2. Load Adapter: Use PeftModel.from_pretrained(base_model, "path/to/adapter").
   3. Merge: Run model = model.merge_and_unload(). This bakes the LoRA weights into the base weights, creating a standard dense model.
   4. Inject into HookedTransformer: Use HookedTransformer.from_pretrained(..., hf_model=model).
