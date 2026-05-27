# API Setup

The generator uses an LLM backend for artifact generation and text polishing.
The default public path is Google Gemini through the `google-genai` Python
package.

## Model Provenance

The released Hollywood 200K dataset was generated with the historical model
name `gemini-3.1-flash-lite-preview`. That preview endpoint has been
discontinued. For new generation runs, this release uses
`gemini-3.1-flash-lite` as the default Gemini model in
`generator/model_defaults.py` and `.env.example`.

This means published data provenance and current reproducibility setup differ
only in the model endpoint name:

```text
released dataset: gemini-3.1-flash-lite-preview
new runs:         gemini-3.1-flash-lite
```

## Gemini

1. Create a Google AI Studio API key.
2. Copy `.env.example` to `.env`.
3. Set:

```text
LLM_PROVIDER=gemini
GOOGLE_API_KEY=<your key>
```

4. Run:

```bash
python generator/smoke_gemini_provider.py
```

The default model roles are defined in `generator/model_defaults.py`. Override
them with environment variables when needed:

```text
MIRAGE_DEFAULT_GEMINI_MODEL=gemini-3.1-flash-lite
MIRAGE_DEFAULT_GEMINI_PRO_MODEL=gemini-3.1-flash-lite
MIRAGE_MODEL_ARTIFACT_BULK=gemini-3.1-flash-lite
MIRAGE_MODEL_ARTIFACT_PRO=gemini-3.1-flash-lite
```

`MIRAGE_MODEL_<ROLE>` overrides one role, while `MIRAGE_DEFAULT_GEMINI_MODEL`
and `MIRAGE_DEFAULT_GEMINI_PRO_MODEL` override the default non-Pro and Pro role
groups. The `MIRAGE_` prefix is historical and kept for compatibility with the
generation code.

## OpenAI-Compatible HTTP Backend

The repository does not start or manage local model weights. To use a local or
hosted non-Gemini model, run an OpenAI-compatible HTTP backend separately and
point the generator at it. This includes vLLM, Ollama, Text Generation
Inference, LiteLLM, the OpenAI API, or another service exposing
`/v1/chat/completions`.

```text
LLM_PROVIDER=local
LOCAL_LLM_URL=http://127.0.0.1:8000/v1
LOCAL_LLM_MODEL=<served model name>
LOCAL_LLM_API_KEY=not-needed
```

Check endpoint connectivity without running the full pipeline:

```bash
python generator/check_local_llm.py --model <served model name>
```

Do not commit `.env` or API keys.
