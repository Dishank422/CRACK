cs:
	flake8 .
black:
	black .


install:
	uv sync --all-groups

build:
	uv build

test:
	pytest --log-cli-level=INFO
tests: test

# Generate CLI reference documentation
# Does not work on Windows due to PYTHONUTF8 env var setting
cli-reference:
	PYTHONUTF8=1 typer CRACK.cli utils docs --name CRACK --title="<a href=\"https://github.com/Dishank422/CRACK\"><img src=\"https://raw.githubusercontent.com/Dishank422/History-Helper-Privacy-Policy/main/CRACK-bot-1_64top.png\" align=\"left\" width=64 height=50 title=\"CRACK: AI Code Reviewer\"></a>CRACK CLI Reference" --output documentation/command_line_reference.md
cli-ref: cli-reference
cli-doc: cli-reference
cli-docs: cli-reference