"""Test Gemini with different thinking-disable syntaxes."""
import asyncio
from dotenv import load_dotenv
import litellm

load_dotenv()

async def try_variant(label, **kwargs):
    try:
        resp = await litellm.acompletion(
            model="gemini/gemini-2.5-pro",
            messages=[{"role": "user", "content": "Write a one-line Python hello world."}],
            max_tokens=100,
            temperature=0.7,
            **kwargs,
        )
        content = resp.choices[0].message.content
        usage = resp.usage
        reasoning = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
        text = getattr(usage.completion_tokens_details, "text_tokens", 0) or 0
        cost = getattr(resp, "_hidden_params", {}).get("response_cost", 0.0) or 0.0
        print(f"[{label}]")
        print(f"  content: {repr(content)[:80]}")
        print(f"  reasoning_tokens: {reasoning}, text_tokens: {text}")
        print(f"  cost: ${cost:.6f}")
        print()
    except Exception as e:
        print(f"[{label}] ERROR: {type(e).__name__}: {str(e)[:150]}\n")

async def main():
    await try_variant("1. no config (baseline)")
    await try_variant("2. reasoning_effort=minimal", reasoning_effort="minimal")
    await try_variant("3. reasoning_effort=none", reasoning_effort="none")
    await try_variant(
        "4. thinking={type:disabled}",
        thinking={"type": "disabled"},
    )
    await try_variant(
        "5. thinking={type:disabled,budget_tokens:0}",
        thinking={"type": "disabled", "budget_tokens": 0},
    )

if __name__ == "__main__":
    asyncio.run(main())
