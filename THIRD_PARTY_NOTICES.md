# Third-party notices

## BrowserGym

This project includes a limited, unmodified source subset from BrowserGym:

- Copyright 2024 ServiceNow
- Repository: https://github.com/ServiceNow/BrowserGym
- Commit: `9e779f087de9a65668b6974d11f9ce9816026e96`
- License: Apache License 2.0

The copied source, dependency-closure rationale, and byte-level checksums are
documented in `vendor/browsergym/SOURCE.md`. The upstream license is reproduced
in `vendor/browsergym/LICENSE`.

## OpenAI Agents SDK

This project depends on `openai-agents==0.18.3` and uses its structured function-tool
runner, model input filter, and verified tool-to-final-output hook.

- Copyright OpenAI
- Repository: https://github.com/openai/openai-agents-python
- Version: `0.18.3`
- License: MIT

The package and its license are installed from PyPI; its source is not vendored here.

## Design references without copied source

The BrowserActor lifecycle follows publicly documented ideas from DeerFlow, while the
tool contracts and verification model were informed by Playwright MCP and Stagehand.
No DeerFlow, Playwright MCP, Stagehand, Browser-Use, Skyvern, cdp-use, or UI-TARS source
is copied into the new `web_agent` implementation.
