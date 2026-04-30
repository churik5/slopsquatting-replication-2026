"""One-shot connectivity test: 1 tiny request to each model."""
import os
import asyncio
from dotenv import load_dotenv
import litellm

load_dotenv()

MODELS = [
    ("anthropic/claude-sonnet-4-5", "ANTHROPIC_API_KEY"),
    ("anthropic/claude-haiku-4-5", "ANTHROPIC_API_KEY"),
    ("openai/gpt-5.4-mini", "OPENAI_API_KEY"),
    ("gemini/gemini-2.5-pro", "GEMINI_API_KEY"),
    ("deepseek/deepseek-chat", "DEEPSEEK_API_KEY"),
]

async def ping(model_id: str, env_var: str):
    if not os.getenv(env_var):
        return model_id, "MISSING_KEY", 0.0
    try:
        extra = {}
        if "gemini" in model_id:
            extra["reasoning_effort"] = "minimal"
        resp = await litellm.acompletion(
            model=model_id,
            messages=[{"role": "user", "content": "Say 'ok' in one word."}],
            max_tokens=50,
            temperature=0,
            **extra,
        )
        text = resp.choices[0].message.content
        cost = getattr(resp, "_hidden_params", {}).get("response_cost", 0.0) or 0.0
        return model_id, f"OK: {(text or '').strip()[:30]}", cost
    except Exception as e:
        return model_id, f"ERROR: {type(e).__name__}: {str(e)[:100]}", 0.0

async def main():
    print("Pinging 5 models with minimal requests...\n")
    results = await asyncio.gather(*[ping(m, k) for m, k in MODELS])
    total_cost = 0.0
    for model_id, status, cost in results:
        marker = "✅" if status.startswith("OK") else "❌"
        print(f"{marker} {model_id:40} | ${cost:.6f} | {status}")
        total_cost += cost
    print(f"\nTotal cost: ${total_cost:.6f}")

if __name__ == "__main__":
    asyncio.run(main())
