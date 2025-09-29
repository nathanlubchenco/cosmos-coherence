#!/usr/bin/env python3
"""Add noqa: E501 to all long lines in halueval_prompts.py"""


file_path = "src/cosmos_coherence/benchmarks/implementations/halueval_prompts.py"

with open(file_path, "r") as f:
    lines = f.readlines()

# Add noqa: E501 to lines that are too long and don't already have it
new_lines = []
for line in lines:
    # Skip if already has noqa
    if "# noqa: E501" in line:
        new_lines.append(line)
    # Add noqa if line is too long (>100 chars) and doesn't have it
    elif len(line.rstrip()) > 100:
        # If line ends with """ or other quote, add before
        if line.rstrip().endswith('"""'):
            new_lines.append(line)
        else:
            new_lines.append(line.rstrip() + "  # noqa: E501\n")
    else:
        new_lines.append(line)

with open(file_path, "w") as f:
    f.writelines(new_lines)

print(f"Fixed {file_path}")
