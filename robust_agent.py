"""Day 6: Robust agent with tool validation and retry."""

import json

from config import client, MODEL, banner
from tools_v2 import TOOLS, TOOL_FUNCTIONS, SCHEMAS
from validate import validate_arguments


SYSTEM_PROMPT = (
    "You are a college fee assistant. Never guess a fee: always call "
    "get_course_fee. Use calculator for every arithmetic step. "
    "Valid course codes: CS101, AI202, DS303. "
    "If no tool is needed, answer directly."
)

MAX_TOKENS = 500
REPEAT_LIMIT = 3


def handle_tool_call(call, log=True):
    """Safely execute one tool call and always return a string."""

    name = call.function.name
    raw = call.function.arguments or "{}"

    # 1. Check JSON
    try:
        arguments = json.loads(raw)
    except json.JSONDecodeError as error:
        return (
            f"Argument error: invalid JSON ({error}). "
            f"Send valid JSON for '{name}'."
        )

    # 2. Check whether tool exists
    function = TOOL_FUNCTIONS.get(name)

    if function is None:
        return (
            f"Unknown tool: {name}. "
            f"Available tools: {', '.join(TOOL_FUNCTIONS)}."
        )

    # 3. Validate arguments against schema
    problem = validate_arguments(arguments, SCHEMAS[name])

    if problem:
        return f"Argument error: {problem}"

    # 4. Run the tool safely
    try:
        result = str(function(**arguments))
    except Exception as error:
        result = (
            f"Tool error in {name}: "
            f"{type(error).__name__}: {error}"
        )

    if log:
        print(f"      {name}({arguments}) -> {result[:100]}")

    return result


def agent(question, max_steps=6, verbose=True):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": question
        }
    ]

    seen = {}
    max_tokens = MAX_TOKENS

    for step in range(1, max_steps + 1):

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                temperature=0,
                max_tokens=max_tokens
            )

        except Exception as error:
            return (
                f"API error: {type(error).__name__}: {error}"
            )

        choice = response.choices[0]
        message = choice.message

        # Handle truncated response
        if choice.finish_reason == "length":

            if max_tokens >= 2000:
                return (
                    "Stopped: the reply was still truncated "
                    "at 2000 tokens."
                )

            max_tokens *= 2

            if verbose:
                print(
                    f"   step {step}: truncated, "
                    f"retrying with max_tokens={max_tokens}"
                )

            continue

        # Final answer
        if not message.tool_calls:
            return (message.content or "").strip()

        # Add assistant tool-call message
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments
                        }
                    }
                    for call in message.tool_calls
                ]
            }
        )

        if verbose:
            print(
                f"   step {step}: "
                f"{len(message.tool_calls)} tool call(s)"
            )

        # Process every tool call
        for call in message.tool_calls:

            signature = (
                call.function.name,
                call.function.arguments
            )

            seen[signature] = seen.get(signature, 0) + 1

            if seen[signature] >= REPEAT_LIMIT:
                return (
                    f"Stopped: {call.function.name} was called "
                    f"{REPEAT_LIMIT} times with the same arguments "
                    f"and made no progress."
                )

            result = handle_tool_call(
                call,
                log=verbose
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result
                }
            )

    return "Stopped: maximum steps reached without a final answer."


if __name__ == "__main__":

    print("=" * 70)
    print(banner)
    print("=" * 70)

    questions = [
        "What is the total fee for CS101 and AI202 after a 10% scholarship?",
        "Is DS303 more expensive than CS101, and by how much?",
        "What is the fee for ME404?",
        "Write a one-line welcome message for new students."
    ]

    for question in questions:
        print("\nQ:", question)
        print("A:", agent(question))