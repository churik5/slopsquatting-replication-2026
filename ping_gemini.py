"""Targeted Gemini ping with realistic prompt."""
import asyncio
from dotenv import load_dotenv
import litellm

load_dotenv()

async def main():
    try:
        resp = await litellm.acompletion(
            model="gemini/gemini-2.5-pro",
            messages=[{"role": "user", "content": "Write a one-line Python hello world."}],
            max_tokens=50,
            temperature=0.7,
        )
        print("Full response object:")
        print(resp)
        print("\n--- content ---")
        print(repr(resp.choices[0].message.content))
        print("\n--- finish_reason ---")
        print(resp.choices[0].finish_reason)
        cost = getattr(resp, "_hidden_params", {}).get("response_cost", 0.0) or 0.0
        print(f"\nCost: ${cost:.6f}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
