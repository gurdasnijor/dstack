---
name: dstack-prototyping
description: |
  Use with the dstack skill for model-serving work of any modality when the image, serving command, resources, backend/fleet choice, or service behavior is not proven. Guides task-first prototyping on real hardware and verification through the final dstack service URL.
---

# dstack Prototyping

Use `/dstack` for CLI commands, YAML fields, apply/attach behavior, service URLs,
and other dstack syntax. This skill explains how to use dstack runs while the
model-serving configuration is still unknown.

## Goal

Find a working dstack service configuration for the requested model and its real
API, whether it generates tokens, vectors, images, video, audio, or another
documented output.

Before submitting a service, use a task on real hardware to test the serving
image, install/runtime assumptions, model download, cache path, command, port,
launch flags, resources, env vars, backend/fleet choice, and local model
request. Then submit the same configuration as a service and verify the model
through the dstack service URL.

## Choose Where To Run

Choose only VM-based backends, SSH fleets, or Kubernetes fleets because they support idle instances and/or instance volumes. That lets later runs reuse the provisioned/idle instance or instance volumes used by runs for caching model weights (and possibly other writes). You must follow this rule even if there are fleets/backends/offers that are cheaper. The only exception from this rule is when the required GPU class (regardless of the price) is not available through VM-based backend, SSH fleet, or Kubernetes fleet.

Read `https://dstack.ai/docs/concepts/backends.md` to know exactly which
backends are VM-based.

## Check Serving Sources

Check serving-framework sources early enough to choose the image, command,
launch flags, resources, cache paths, request format, and expected model
behavior.

For vLLM and SGLang, use these as credible sources:

- vLLM recipes and model index: `https://recipes.vllm.ai/` and
  `https://recipes.vllm.ai/models.json`
- vLLM recipe docs: `https://docs.vllm.ai/projects/recipes/en/stable/`
- SGLang docs and cookbook: `https://docs.sglang.ai/` and
  `https://lmsysorg.mintlify.app/cookbook/intro`

Use deeper serving-engine writeups, such as
`https://www.lmsys.org/blog/2026-07-02-agent-assisted-sglang-development`, when
these references do not explain the model, hardware, or serving failure.

Do not assume vLLM or SGLang is the only valid serving stack. Inspect model
source metadata first, then evaluate runtimes that explicitly support the
detected architecture and modality. For Hugging Face diffusion repositories,
inspect `model_index.json`, component configs, and the model card. vLLM-Omni's
Diffusers adapter is a candidate for compatible pipelines and exposes
`POST /v1/images/generations`; verify current vLLM-Omni documentation before
choosing its image and command. ComfyUI, Diffusers services, media runtimes, or a
model-specific HTTP server are valid when they provide a stable API and probe.

## Use A Task Before Service

Before submitting a service, start a long-lived task:

```yaml
commands:
  - sleep infinity
```

or an equivalent idle command.

Submit the task detached, attach or SSH into it when available, and run commands
inside the live environment. Test the image, installs, model download and cache
path, serving command, port, launch flags, local model request, and expected
model behavior.

When starting a long-running command in the background from a non-interactive
SSH command, use `nohup`, redirect stdin from `/dev/null`, and redirect
stdout/stderr to a log file so the SSH command returns while the process keeps
running. For example (the command can be any long-running command):

```shell
nohup vllm serve ... </dev/null > /tmp/vllm.log 2>&1 &
```

If the image, hardware choice, or major install path changes, submit another
task so the changed setup is tested before service verification.

Do not move to a service after checking only GPU visibility, imports, logs, or a
health endpoint. Start the server inside the task and send a request that uses
the requested model. For a chat or reasoning model, check the response behavior
the endpoint is expected to support, such as reasoning output when that model is
supposed to expose it.

Follow `/dstack` structured status guidance when polling task or service status.
After requesting a task or service stop before another submission, wait until
that run reaches a terminal status. This allows dstack to reuse its instance or
instance volumes when available.

## Verify As A Service

Submit the service after the task has verified the configuration: image,
command, port, resources, env vars, cache mounts if used, backend/fleet choice,
and model request.

Use the service as a duplicate check of the same configuration under dstack
service runtime. The model request that worked locally in the task must also work
through the dstack service URL.

If service verification fails because the image, install, model download,
command, resources, cache, or model behavior needs to change, go back to a task.
If the tested serving setup is still right and only the dstack service
configuration is wrong, fix the configuration and submit the service again.

For non-chat services, omit dstack's chat-only `model` field and configure an
explicit HTTP health probe. This keeps the service generic while dstack still
owns placement, lifecycle, proxying, and deployment.

## Benchmark The Real Output

Benchmark through the final dstack service URL, not an SSH tunnel or task-local
port. Exclude warmups. Validate the output, not only the HTTP status: token
content for generation, vector shape for embeddings, decoded media for
image/audio/video, or the documented response contract for a custom API.

For an OpenAI-compatible image endpoint, use the packaged
`scripts/benchmark_images.py`. Give it a JSON request body with
`response_format: b64_json`; it sends a warmup plus measured requests, validates
that each response contains decodable PNG/JPEG/WebP bytes with the requested
dimensions, and writes a benchmark object accepted by the endpoint preset
schema. Run `python scripts/benchmark_images.py --help` for options.
