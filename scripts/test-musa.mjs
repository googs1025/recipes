import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";

import yaml from "js-yaml";

import {
  buildDockerArgv,
  buildDockerRun,
  computeDockerMeta,
  resolveCommand,
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

test("taxonomy defines a restricted 2x MTT S5000 profile", () => {
  const taxonomy = yaml.load(
    fs.readFileSync(new URL("../taxonomy.yaml", import.meta.url), "utf8"),
  );
  const profile = taxonomy.hardware_profiles.mtt_s5000_2x;

  assert.deepEqual(profile, {
    brand: "MooreThreads",
    generation: "musa",
    display_name: "MTT S5000 2x",
    description: "Moore Threads MTT S5000 · 2× 80 GB VRAM · MUSA backend",
    gpu_count: 2,
    vram_gb: 160,
    multi_node: false,
    scalable: false,
    restricted: true,
  });
});

test("Qwen3.5-27B renders its documented 2x S5000 serving profile", () => {
  const readYaml = (relativePath) =>
    yaml.load(
      fs.readFileSync(new URL(`../${relativePath}`, import.meta.url), "utf8"),
    );
  const taxonomy = readYaml("taxonomy.yaml");
  const recipe = readYaml("models/Qwen/Qwen3.5-27B.yaml");
  const strategy = readYaml("strategies/single_node_tp.yaml");
  const strategies = { single_node_tp: strategy };
  const result = resolveCommand(
    recipe,
    "default",
    "single_node_tp",
    "mtt_s5000_2x",
    ["spec_decoding"],
    strategies,
    taxonomy,
  );
  const flag = (name) => result.argv[result.argv.indexOf(name) + 1];
  const meta = computeDockerMeta(
    recipe,
    recipe.variants.default,
    taxonomy.hardware_profiles.mtt_s5000_2x,
    "mtt_s5000_2x",
  );

  assert.equal(recipe.meta.hardware.mtt_s5000_2x, "supported");
  assert.deepEqual(recipe.hardware_opt_in_features.mtt_s5000_2x, [
    "tool_calling",
    "reasoning",
    "encoder_parallel",
  ]);
  assert.equal(recipe.variants.default.hardware_overrides.mtt_s5000_2x.install.pip, false);
  assert.equal(recipe.variants.fp8.hardware_overrides.mtt_s5000_2x.install.pip, false);
  assert.equal(meta.image, MUSA_IMAGE);
  assert.equal(flag("--tensor-parallel-size"), "2");
  assert.equal(flag("--max-model-len"), "7168");
  assert.equal(flag("--max-num-seqs"), "64");
  assert.equal(flag("--attention-config"), '{"backend":"FLASH_ATTN"}');
  assert.deepEqual(JSON.parse(flag("--speculative-config")), {
    method: "mtp",
    num_speculative_tokens: 1,
  });
  assert.ok(result.argv.includes("--no-enable-prefix-caching"));
  assert.ok(result.argv.includes("--no-enable-chunked-prefill"));
  assert.equal(result.env.VLLM_PLUGINS, "musa,musa_custom_ops");
  assert.equal(result.env.VLLM_WORKER_MULTIPROC_METHOD, "spawn");
  assert.equal(result.env.VLLM_DISABLE_COMPILE_CACHE, "1");
  assert.equal(result.env.SAFETENSORS_FAST_GPU, "1");
});
