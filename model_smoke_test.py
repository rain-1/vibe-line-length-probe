from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    model_id = "google/gemma-3-270m"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="cpu")

    prompt = "Write a short two-line poem about the sea breeze:"
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    generated = model.generate(
        **encoded,
        max_new_tokens=30,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )

    print(tokenizer.decode(generated[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
