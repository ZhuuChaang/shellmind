def extract_json(text: str) -> str:
    """
    Extract the first valid JSON object from a string.
    Returns the substring from first { to the matching }.
    Raises ValueError if no JSON object is found.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found")

    # 简单的括号匹配计数
    brace_count = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                return text[start:i+1]

    raise ValueError("No complete JSON object found")
