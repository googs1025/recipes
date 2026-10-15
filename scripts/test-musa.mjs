import assert from "node:assert/strict";
import test from "node:test";

import {
  buildDockerArgv,
  buildDockerRun,
  computeDockerMeta,
} from "../src/lib/command-synthesis.js";

const MUSA_IMAGE =
  "registry.mthreads.com/mcconline/inference/vllm/vllm-openai:v0.28.0";
const MUSA_PROFILE = {
  brand: "MooreThreads",
  generation: "musa",
  gpu_count: 2,
};

function musaMeta(recipe = { model: {} }, variant = {}, hwId = "mtt_s5000_2x") {
  return computeDockerMeta(recipe, variant, MUSA_PROFILE, hwId);
}

test("MUSA metadata resolves the public vLLM-MUSA image and runtime", () => {
  const meta = musaMeta();

  assert.equal(meta.image, MUSA_IMAGE);
  assert.equal(meta.brandKey, "musa");
  assert.equal(meta.isMusa, true);
  assert.match(meta.gpuFlags, /MTHREADS_VISIBLE_DEVICES=all/);
  assert.doesNotMatch(meta.gpuFlags, /--gpus all/);
});

test("MUSA image overrides retain exact-hardware then model precedence", () => {
  const modelImage = "registry.example.com/vllm-musa:model";
  const exactImage = "registry.example.com/vllm-musa:exact";
  const recipe = { model: { docker_image: { musa: modelImage } } };

  assert.equal(musaMeta(recipe).image, modelImage);
  assert.equal(
    musaMeta(recipe, {
      hardware_overrides: {
        mtt_s5000_2x: { docker_image: exactImage },
      },
    }).image,
    exactImage,
  );
});

test("MUSA shell and argv Docker commands use host networking", () => {
  const meta = musaMeta();
  const command = "vllm serve Qwen/Qwen3.5-27B --tensor-parallel-size 2";
  const argv = [
    "vllm",
    "serve",
    "Qwen/Qwen3.5-27B",
    "--tensor-parallel-size",
    "2",
  ];
  const env = { MUSA_VISIBLE_DEVICES: "0,1" };
  const shell = buildDockerRun({
    command,
    env,
    image: meta.image,
    gpuFlags: meta.gpuFlags,
    isMusa: meta.isMusa,
  });
  const dockerArgv = buildDockerArgv({ argv, env, meta });

  for (const expected of [
    "MTHREADS_VISIBLE_DEVICES=all",
    "MUSA_VISIBLE_DEVICES=0,1",
    "--privileged",
    "--net=host",
    "--shm-size=16g",
  ]) {
    assert.match(shell, new RegExp(expected.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
  }
  assert.doesNotMatch(shell, /--gpus all/);
  assert.doesNotMatch(shell, /-p 8000:8000/);

  assert.deepEqual(dockerArgv.slice(0, 10), [
    "docker",
    "run",
    "-e",
    "MTHREADS_VISIBLE_DEVICES=all",
    "--privileged",
    "--net=host",
    "--shm-size",
    "16g",
    "-v",
    "~/.cache/huggingface:/root/.cache/huggingface",
  ]);
  assert.ok(dockerArgv.includes("MUSA_VISIBLE_DEVICES=0,1"));
  assert.ok(!dockerArgv.includes("--gpus"));
  assert.ok(!dockerArgv.includes("-p"));
});

test("NVIDIA Docker metadata remains unchanged", () => {
  const meta = computeDockerMeta(
    { model: {} },
    {},
    { brand: "NVIDIA", generation: "hopper", gpu_count: 8 },
    "h200",
  );

  assert.equal(meta.image, "vllm/vllm-openai:latest");
  assert.equal(meta.brandKey, "nvidia");
  assert.equal(meta.gpuFlags, "--gpus all");
  assert.equal(meta.isMusa, false);
});
