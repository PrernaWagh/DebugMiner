"""Convert Python traceback text into a cautious, user-provided bug transaction."""
import re

ERROR_GUIDANCE = {
    "KeyError": ("A dictionary lookup requested a key that was not present.",
                 "Check key spelling/type. Use .get() for optional keys.",
                 "value = data.get('name')"),
    "IndexError": ("A sequence was accessed at an index that does not exist.",
                   "Check sequence length and index bounds.",
                   "if 0 <= i < len(items):\n    value = items[i]"),
    "TypeError": ("An operation received a value of an unsuitable type.",
                  "Check types and function arguments.",
                  "print(type(value))"),
    "NameError": ("Python could not find a variable or name in scope.",
                  "Check spelling, initialization, and scope.",
                  "count = 0\nprint(count)"),
    "AttributeError": ("An object does not have the requested attribute.",
                       "Check the object's type and whether it is None.",
                       "if value is not None:\n    print(type(value))"),
    "ValueError": ("A function received a value of acceptable type but invalid content.",
                   "Validate input before conversion.",
                   "text = '123'\nif text.isdigit():\n    n = int(text)"),
    "ZeroDivisionError": ("Division by zero.", "Check the denominator before dividing.",
                          "if d != 0:\n    r = n / d"),
    "ImportError": ("Could not import a module or name.",
                    "Check package name, environment, and version.", "# verify import path"),
    "ModuleNotFoundError": ("Python could not find a module.",
                            "Install it in the active environment.", "python -m pip install package"),
    "FileNotFoundError": ("The requested file was not found.",
                          "Check the path and working directory.",
                          "from pathlib import Path\nprint(Path('data.txt').exists())"),
    "SyntaxError": ("Python could not parse the source code.",
                    "Check colons, brackets, quotes, and indentation.",
                    "if condition:\n    print('fixed')"),
}

def identify_error(error_message):
    message = error_message or ""
    for name in sorted(ERROR_GUIDANCE, key=len, reverse=True):
        if re.search(r"\b" + re.escape(name) + r"\b", message):
            return name
    match = re.search(r"\b([A-Za-z_]\w*(?:Error|Exception))\b", message)
    return match.group(1) if match else "Unknown"

def get_guidance(error_type):
    if error_type in ERROR_GUIDANCE:
        explanation, fix, example = ERROR_GUIDANCE[error_type]
        return {"explanation": explanation, "fix": fix, "example": example}
    return {
        "explanation": "The exception name is unknown; the cause cannot be determined from the name alone.",
        "fix": "Inspect the final traceback lines and reproduce the error in a small test.",
        "example": "# inspect the failing line\n# create a minimal reproducible test",
    }

def build_transaction_from_error(error_type, project="", python_version=""):
    items = []
    if error_type and error_type != "Unknown":
        items.append(f"Fault_Acronym={error_type}")
    if project and project.strip():
        items.append(f"Project={project.strip()}")
    if python_version and python_version.strip():
        version = python_version.strip()
        parts = version.split(".")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            version = f"{parts[0]}.{parts[1]}"
        items.append(f"PythonVersion={version}")
    return items
