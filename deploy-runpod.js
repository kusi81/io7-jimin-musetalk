/**
 * One-shot RunPod Serverless Template + Endpoint deployer for IO7 Sales (Kim Ji-min Gwazang).
 *
 * Usage:
 *   node lipsync-worker/deploy-runpod.js --image ghcr.io/kusi81/io7-jimin-musetalk:v1 [--gpu L4|4090] [--name io7-jimin-musetalk]
 *
 * Reads RUNPOD_API_KEY from io7-sales-agent/.env (or process.env.RUNPOD_API_KEY),
 * creates the Serverless template & Flex endpoint (min=0 idle $0, FlashBoot ON),
 * and writes RUNPOD_ENDPOINT_ID into .env automatically.
 */

import 'dotenv/config';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const ENV_PATH = path.join(ROOT, '.env');
const REST_BASE = 'https://rest.runpod.io/v1';

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith('--')) {
      const k = argv[i].slice(2);
      const v = argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[++i] : 'true';
      out[k] = v;
    }
  }
  return out;
}

async function runpodFetch(apiKey, endpointPath, method = 'GET', body = null) {
  const res = await fetch(`${REST_BASE}${endpointPath}`, {
    method,
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  let json;
  try {
    json = JSON.parse(text);
  } catch {
    json = { raw: text };
  }
  if (!res.ok) {
    throw new Error(`RunPod API ${method} ${endpointPath} failed (${res.status}): ${JSON.stringify(json)}`);
  }
  return json;
}

function upsertEnv(filePath, updates) {
  let content = fs.existsSync(filePath) ? fs.readFileSync(filePath, 'utf8') : '';
  for (const [key, val] of Object.entries(updates)) {
    const line = `${key}=${val}`;
    const re = new RegExp(`^${key}=.*$`, 'm');
    if (re.test(content)) {
      content = content.replace(re, line);
    } else {
      if (content.length > 0 && !content.endsWith('\n')) content += '\n';
      content += `${line}\n`;
    }
  }
  fs.writeFileSync(filePath, content, 'utf8');
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const apiKey = (process.env.RUNPOD_API_KEY || '').trim();
  if (!apiKey) {
    console.error('[ERROR] RUNPOD_API_KEY is not set in io7-sales-agent/.env');
    process.exit(1);
  }

  const imageName = args.image || process.env.RUNPOD_IMAGE_NAME;
  if (!imageName) {
    console.error('[ERROR] Missing --image <registry/repo:tag> (e.g. ghcr.io/kusi81/io7-jimin-musetalk:v1)');
    process.exit(1);
  }

  const gpuPref = (args.gpu || 'L4').toUpperCase();
  const gpuTypeIds =
    gpuPref === '4090'
      ? ['NVIDIA GeForce RTX 4090', 'NVIDIA L4']
      : ['NVIDIA L4', 'NVIDIA GeForce RTX 4090'];

  const stamp = new Date().toISOString().slice(0, 10).replace(/-/g, '');
  const baseName = args.name || 'io7-jimin-musetalk';
  const templateName = `${baseName}-tpl-${stamp}`;
  const endpointName = `${baseName}-ep`;

  console.log(`[1/3] Creating RunPod Serverless Template "${templateName}" -> ${imageName}`);
  const tpl = await runpodFetch(apiKey, '/templates', 'POST', {
    name: templateName,
    imageName,
    category: 'NVIDIA',
    isServerless: true,
    containerDiskInGb: 25,
    volumeInGb: 0,
    isPublic: false,
    env: {
      EAGER_WARMUP: '1',
      LIPSYNC_FPS: '25',
      LIPSYNC_PATCH_FORMAT: 'webp',
      LIPSYNC_PATCH_QUALITY: '82',
    },
  });
  const templateId = tpl.id;
  if (!templateId) throw new Error(`Template created without id: ${JSON.stringify(tpl)}`);
  console.log(`      Template ID: ${templateId}`);

  console.log(`[2/3] Creating RunPod Serverless Flex Endpoint "${endpointName}" (${gpuTypeIds.join(', ')})`);
  const ep = await runpodFetch(apiKey, '/endpoints', 'POST', {
    name: endpointName,
    templateId,
    computeType: 'GPU',
    gpuCount: 1,
    gpuTypeIds,
    workersMin: 0,
    workersMax: 2,
    idleTimeout: 20,
    scalerType: 'QUEUE_DELAY',
    scalerValue: 2,
    flashboot: true,
  });
  const endpointId = ep.id;
  if (!endpointId) throw new Error(`Endpoint created without id: ${JSON.stringify(ep)}`);
  console.log(`      Endpoint ID: ${endpointId}`);

  console.log(`[3/3] Writing RUNPOD_ENDPOINT_ID & LIPSYNC_GPU_TIER to ${ENV_PATH}`);
  upsertEnv(ENV_PATH, {
    RUNPOD_ENDPOINT_ID: endpointId,
    LIPSYNC_GPU_TIER: gpuPref === '4090' ? '4090' : 'L4',
    LIPSYNC_CHUNK_MS: '480',
    LIPSYNC_TIMEOUT_MS: '2800',
  });

  console.log(`\n[DONE] RunPod Serverless Endpoint ready!`);
  console.log(`  Endpoint ID : ${endpointId}`);
  console.log(`  GPU Tier    : ${gpuPref} (min=0 idle $0, max=2, idleTimeout=20s, FlashBoot=ON)`);
  console.log(`  Next Step   : Restart IO7 Sales server and click "Pre-warm 예열" in MP4 클립 관리 tab.`);
}

main().catch((err) => {
  console.error('[DEPLOY FAILED]', err.message);
  process.exit(1);
});
