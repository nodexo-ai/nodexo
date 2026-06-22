#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import readline from "node:readline/promises";

const DEFAULT_API = process.env.NODEXO_API_URL || "https://nodexo.ai/api";
const DEFAULT_KEY_ENV = "NODEXO_EVM_PRIVATE_KEY";
const DEFAULT_API_KEY_ENV = "NODEXO_API_KEY";
const DEFAULT_STORAGE_GB = 30;

function usage(exitCode = 0) {
  const out = exitCode === 0 ? process.stdout : process.stderr;
  out.write(`Nodexo public API helper

Usage:
  node scripts/public-cli.mjs inventory [--api URL] [--json]
  node scripts/public-cli.mjs marketplace [--api URL] [--json]
  node scripts/public-cli.mjs tiers [--api URL] [--json]
  node scripts/public-cli.mjs quote --gpu A6000 --duration 1h --storage-gb 30 --ssh-key ~/.ssh/id_ed25519.pub [--api URL]
  node scripts/public-cli.mjs rent --gpu A6000 --duration 1h --storage-gb 30 --ssh-key ~/.ssh/id_ed25519.pub [--api URL]
  node scripts/public-cli.mjs rent --payment credits --gpu A6000 --storage-gb 30 --ssh-key ~/.ssh/id_ed25519.pub [--api URL]
  node scripts/public-cli.mjs credits [--api-key KEY] [--json]
  node scripts/public-cli.mjs account-rentals [--api-key KEY] [--json]
  node scripts/public-cli.mjs account-ssh-keys [--api-key KEY] [--json]
  node scripts/public-cli.mjs account-ssh-key-add --ssh-key ~/.ssh/id_ed25519.pub [--label NAME] [--api-key KEY]
  node scripts/public-cli.mjs account-ssh-key-remove --key-id ID [--api-key KEY]
  node scripts/public-cli.mjs get --rental-id ID --recovery-token TOKEN [--json]
  node scripts/public-cli.mjs end --rental-id ID --recovery-token TOKEN [--json]
  node scripts/public-cli.mjs extend --rental-id ID --hours 4 --recovery-token TOKEN [--json]
  node scripts/public-cli.mjs ssh-key-add --rental-id ID --recovery-token TOKEN --ssh-key ~/.ssh/id_ed25519.pub
  node scripts/public-cli.mjs ssh-key-remove --rental-id ID --recovery-token TOKEN --ssh-pub-key "ssh-ed25519 AAAA..."

Environment:
  NODEXO_API_URL       Public API base, default ${DEFAULT_API}
  ${DEFAULT_KEY_ENV}        EVM private key used for x402 signing
  ${DEFAULT_API_KEY_ENV}      Account API key used for credit rentals and account reads
  NODEXO_RENTAL_RECOVERY_TOKEN Default recovery token for get/end/extend/ssh-key commands

Notes:
  During public testnet preview, non-admin rent/extend/top-up commands stop at
  the launch gate before funds move or containers start.
  x402 rent/extend uses v2 upto payments on the public /api routes.
  credit rent uses account balance and requires an account API key.
  quote performs the first unsigned request and prints the x402 payment requirements.
`);
  process.exit(exitCode);
}

const COLORS = process.stdout.isTTY && !process.env.NO_COLOR;
const ANSI = {
  reset: COLORS ? "\x1b[0m" : "",
  bold: COLORS ? "\x1b[1m" : "",
  dim: COLORS ? "\x1b[2m" : "",
  cyan: COLORS ? "\x1b[36m" : "",
  green: COLORS ? "\x1b[32m" : "",
  yellow: COLORS ? "\x1b[33m" : "",
  red: COLORS ? "\x1b[31m" : "",
  white: COLORS ? "\x1b[37m" : "",
};

function color(text, ...codes) {
  return `${codes.map((code) => ANSI[code] || "").join("")}${text}${ANSI.reset}`;
}

function stripAnsi(value) {
  return String(value).replace(/\x1b\[[0-9;]*m/g, "");
}

function pad(value, width, align = "left") {
  const raw = String(value);
  const extra = Math.max(0, width - stripAnsi(raw).length);
  return align === "right" ? `${" ".repeat(extra)}${raw}` : `${raw}${" ".repeat(extra)}`;
}

function money(value) {
  const n = Number(value || 0);
  return `$${n.toFixed(n >= 10 ? 2 : 2)}`;
}

function percent(value) {
  if (value == null || !Number.isFinite(Number(value))) return "--";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function rangeMaxGb(value) {
  const matches = String(value || "").match(/\d+(?:\.\d+)?/g);
  if (!matches?.length) return null;
  return Math.max(...matches.map(Number));
}

function hoursLabel(hours) {
  const h = Number(hours || 0);
  if (h === 168) return "7d";
  if (h % 24 === 0 && h >= 24) return `${h / 24}d`;
  return `${h}h`;
}

function secondsLabel(seconds) {
  const s = Math.max(0, Math.floor(Number(seconds || 0)));
  if (s >= 86400) {
    const d = Math.floor(s / 86400);
    const h = Math.floor((s % 86400) / 3600);
    return h ? `${d}d ${h}h` : `${d}d`;
  }
  if (s >= 3600) {
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    return m ? `${h}h ${m}m` : `${h}h`;
  }
  if (s >= 60) {
    const m = Math.floor(s / 60);
    const rest = s % 60;
    return rest ? `${m}m ${rest}s` : `${m}m`;
  }
  return `${s}s`;
}

function timeLabel(value) {
  if (!value) return "--";
  const date = typeof value === "number" ? new Date(value * 1000) : new Date(value);
  if (Number.isNaN(date.getTime())) return "--";
  return date.toLocaleString(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function shortAddress(value) {
  const s = String(value || "");
  return s.length > 14 ? `${s.slice(0, 8)}...${s.slice(-6)}` : s || "--";
}

function shortId(value) {
  const s = String(value || "");
  return s.length > 18 ? `${s.slice(0, 10)}...${s.slice(-6)}` : s || "--";
}

function rentalAlias(rentalId) {
  return `nodexo-${String(rentalId || "").slice(0, 12)}`;
}

function sshCommand(conn = {}) {
  if (!conn.ssh_host || !conn.ssh_port) return "--";
  return `ssh -p ${conn.ssh_port} ${conn.ssh_user || "root"}@${conn.ssh_host}`;
}

function printTitle(title, subtitle = "") {
  console.log();
  console.log(`${color("|", "cyan", "bold")} ${color(title, "bold", "white")}`);
  if (subtitle) console.log(`  ${color(subtitle, "dim")}`);
  console.log(color("-".repeat(84), "dim"));
}

function printKv(label, value) {
  console.log(`  ${color(pad(label, 18), "dim")} ${value}`);
}

function parseArgs(argv) {
  const flags = {};
  const rest = [];
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (!arg.startsWith("--")) {
      rest.push(arg);
      continue;
    }
    const eq = arg.indexOf("=");
    if (eq !== -1) {
      flags[arg.slice(2, eq)] = arg.slice(eq + 1);
      continue;
    }
    const key = arg.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith("--")) {
      flags[key] = true;
    } else {
      flags[key] = next;
      i += 1;
    }
  }
  return { flags, rest };
}

function apiBase(flags) {
  const raw = String(flags.api || DEFAULT_API).replace(/\/+$/, "");
  return raw.endsWith("/api") ? raw : `${raw}/api`;
}

function mustString(flags, key, fallback = "") {
  const value = String(flags[key] || fallback || "").trim();
  if (!value) throw new Error(`missing --${key}`);
  return value;
}

function numberFlag(flags, key, fallback = null) {
  const raw = flags[key] ?? fallback;
  if (raw === null || raw === undefined || raw === "") return null;
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new Error(`invalid --${key}: ${raw}`);
  return value;
}

function durationHours(flags, key = "hours", fallback = null) {
  const raw = flags[key] ?? flags.duration ?? fallback;
  if (raw === null || raw === undefined || raw === "") return null;
  const text = String(raw).trim().toLowerCase();
  const match = text.match(/^(\d+)(h|hr|hrs|hour|hours|d|day|days)?$/);
  if (!match) throw new Error(`invalid --${key}: ${raw}`);
  const amount = Number(match[1]);
  const unit = match[2] || "h";
  return unit.startsWith("d") ? amount * 24 : amount;
}

function expandHome(file) {
  if (!file) return file;
  if (file === "~") return os.homedir();
  if (file.startsWith("~/")) return path.join(os.homedir(), file.slice(2));
  return file;
}

function readSshPubKey(flags) {
  const inline = String(flags["ssh-pub-key"] || "").trim();
  if (inline) return inline;
  const explicit = String(flags["ssh-key"] || "").trim();
  const candidates = explicit
    ? [expandHome(explicit)]
    : [path.join(os.homedir(), ".ssh/id_ed25519.pub"), path.join(os.homedir(), ".ssh/id_rsa.pub")];
  for (const candidate of candidates) {
    if (candidate && fs.existsSync(candidate)) {
      return fs.readFileSync(candidate, "utf8").trim();
    }
  }
  throw new Error("missing SSH public key; pass --ssh-key PATH or --ssh-pub-key TEXT");
}

async function readJson(response) {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return { raw: text };
  }
}

async function fetchJson(url, init = {}) {
  const response = await fetch(url, init);
  const body = await readJson(response);
  if (!response.ok) {
    const err = new Error(`${response.status}: ${JSON.stringify(body).slice(0, 500)}`);
    err.status = response.status;
    err.body = body;
    throw err;
  }
  return body;
}

function usdcToUnitsCeil(amount) {
  if (!Number.isFinite(amount) || amount < 0) throw new Error(`invalid USDC amount ${amount}`);
  return BigInt(Math.ceil(amount * 1_000_000));
}

async function createPaidFetch(privateKey, maxPaymentUnits) {
  let x402Client;
  let wrapFetchWithPayment;
  let UptoEvmScheme;
  let privateKeyToAccount;
  try {
    ({ x402Client, wrapFetchWithPayment } = await import("@x402/fetch"));
    ({ UptoEvmScheme } = await import("@x402/evm/upto/client"));
    ({ privateKeyToAccount } = await import("viem/accounts"));
  } catch (error) {
    throw new Error(
      "x402 dependencies are not installed. Run `npm install` in the nodexo repo, then retry.",
    );
  }
  const account = privateKeyToAccount(privateKey.startsWith("0x") ? privateKey : `0x${privateKey}`);
  const signer = {
    address: account.address,
    signTypedData: (message) => account.signTypedData(message),
  };
  const client = new x402Client()
    .register("eip155:*", new UptoEvmScheme(signer))
    .registerPolicy((_version, requirements) =>
      requirements.filter((requirement) => {
        if (requirement.scheme !== "upto") return false;
        try {
          return BigInt(requirement.amount) <= maxPaymentUnits;
        } catch {
          return false;
        }
      }),
    );
  return wrapFetchWithPayment(fetch, client);
}

function privateKeyFromFlags(flags) {
  if (flags["private-key"]) return String(flags["private-key"]).trim();
  const envName = String(flags["private-key-env"] || DEFAULT_KEY_ENV);
  const value = String(process.env[envName] || "").trim();
  if (!value) throw new Error(`missing EVM private key; set ${envName} or pass --private-key-env NAME`);
  return value;
}

function recoveryToken(flags) {
  return mustString(flags, "recovery-token", process.env.NODEXO_RENTAL_RECOVERY_TOKEN || "");
}

function apiKey(flags, required = false) {
  const key = String(flags["api-key"] || process.env[DEFAULT_API_KEY_ENV] || "").trim();
  if (!key && required) throw new Error(`missing account API key; set ${DEFAULT_API_KEY_ENV} or pass --api-key`);
  return key;
}

function authHeaders(flags) {
  const key = apiKey(flags, false);
  if (key) return { authorization: `Bearer ${key}` };
  return { authorization: `Bearer ${recoveryToken(flags)}` };
}

function accountHeaders(flags) {
  return { authorization: `Bearer ${apiKey(flags, true)}` };
}

function paymentMode(flags) {
  const raw = String(flags.payment || flags["payment-mode"] || "").trim().toLowerCase();
  if (flags.credits || raw === "credit") return "credits";
  if (!raw) return "x402";
  if (["x402", "credits"].includes(raw)) return raw;
  throw new Error(`invalid --payment: ${raw}`);
}

function rankOffers(a, b) {
  const priceA = Number(a.price_usdc_per_hour ?? 999999);
  const priceB = Number(b.price_usdc_per_hour ?? 999999);
  const relA = Number(a.reliability_24h ?? 0);
  const relB = Number(b.reliability_24h ?? 0);
  return priceA - priceB || relB - relA || String(a.gpu_model).localeCompare(String(b.gpu_model));
}

async function selectOffer(api, flags) {
  const inventory = await fetchJson(`${api}/inventory`);
  const offers = Array.isArray(inventory.offers) ? inventory.offers : [];
  const gpuNeedle = String(flags.gpu || flags["gpu-model"] || "").trim().toLowerCase();
  const gpuCount = numberFlag(flags, "gpu-count", 1);
  const storageGb = numberFlag(flags, "storage-gb", flags["min-storage-gb"] ?? DEFAULT_STORAGE_GB);
  const memoryGb = numberFlag(flags, "memory-gb", flags["min-ram-gb"] ?? 16);
  let candidates = offers.filter((offer) => offer.rentable && Number(offer.available || 0) > 0);
  if (gpuNeedle) {
    candidates = candidates.filter((offer) => String(offer.gpu_model || "").toLowerCase().includes(gpuNeedle));
  }
  if (gpuCount) {
    candidates = candidates.filter((offer) => Number(offer.gpu_count || 1) === gpuCount);
  }
  if (storageGb) {
    candidates = candidates.filter((offer) => {
      const maxStorage = rangeMaxGb(offer.disk_gb);
      return maxStorage != null && maxStorage >= storageGb;
    });
  }
  if (memoryGb) {
    candidates = candidates.filter((offer) => {
      const maxRam = rangeMaxGb(offer.ram_gb);
      return maxRam != null && maxRam >= memoryGb;
    });
  }
  candidates.sort(rankOffers);
  if (!candidates.length) {
    const matchingListed = offers.filter((offer) => {
      const modelOk = !gpuNeedle || String(offer.gpu_model || "").toLowerCase().includes(gpuNeedle);
      const countOk = !gpuCount || Number(offer.gpu_count || 1) === gpuCount;
      return modelOk && countOk;
    });
    const listed = offers.map((offer) => ({
      gpu_model: offer.gpu_model,
      gpu_count: offer.gpu_count,
      state: offer.state,
      available: offer.available,
      price_usdc_per_hour: offer.price_usdc_per_hour,
    }));
    const err = new Error(
      matchingListed.length
        ? "matching inventory is listed but not currently rentable"
        : "no inventory matched the requested GPU",
    );
    err.inventory = listed;
    err.matchingListed = matchingListed.length;
    throw err;
  }
  return candidates[0];
}

function printTable(rows, headers, write = (line) => console.log(line)) {
  const widths = headers.map((header, i) =>
    Math.max(header.length, ...rows.map((row) => stripAnsi(row[i] ?? "").length)),
  );
  write(headers.map((header, i) => color(pad(header, widths[i]), "dim", "bold")).join("  "));
  write(widths.map((width) => "-".repeat(width)).join("  "));
  for (const row of rows) {
    write(row.map((cell, i) => pad(cell ?? "", widths[i])).join("  "));
  }
}

function offerRows(offers) {
  return offers.map((offer, index) => [
    color(index, "cyan", "bold"),
    `${offer.gpu_count || 1}x ${String(offer.gpu_model || "").replace(/^NVIDIA /, "")}`,
    `${offer.vram_gb || "?"} GB`,
    Number(offer.available || 0) > 0
      ? color(String(offer.available ?? 0), "green", "bold")
      : color(String(offer.available ?? 0), "dim"),
    offer.state === "available"
      ? color("available", "green")
      : color(offer.state || "unavailable", offer.state === "rented" ? "yellow" : "red"),
    offer.attestation || "ZkGEMM",
    `${percent(offer.reliability_24h)} ${offer.reliability_confidence || ""}`.trim(),
    offer.price_usdc_per_hour == null ? "--" : `${money(offer.price_usdc_per_hour)}/hr`,
  ]);
}

async function fetchOffers(api) {
  const data = await fetchJson(`${api}/inventory`);
  return Array.isArray(data.offers) ? data.offers : [];
}

function clearScreen() {
  if (process.stdout.isTTY) {
    process.stdout.write("\x1b[2J\x1b[H");
  }
}

function printRental(data, jsonMode = false) {
  if (jsonMode) {
    console.log(JSON.stringify(data, null, 2));
    return;
  }
  const conn = data.connection || {};
  printTitle("Rental ready", "Save the rental ID and recovery token. The token is required outside a signed-in browser session.");
  printKv("Rental ID", color(data.rental_id, "white", "bold"));
  printKv("Recovery", data.recovery_token ? color("returned", "green", "bold") : color("not returned", "yellow"));
  if (data.gpu) {
    printKv("GPU", `${data.gpu.count || 1}x ${String(data.gpu.model || "").replace(/^NVIDIA /, "")}`);
  }
  if (data.resources) {
    printKv("Resources", `${data.resources.storage_gb || "--"} GB storage · ${data.resources.memory_gb || "--"} GB RAM`);
  }
  if (conn.ssh_host && conn.ssh_port) {
    printKv("SSH", color(sshCommand(conn), "cyan", "bold"));
  }
  if (data.payment) {
    printKv("Payment", `${data.payment.scheme || "x402"} ${data.payment.status || ""}`.trim());
    if (data.payment.max_authorized_usdc != null) {
      printKv("Max authorized", money(data.payment.max_authorized_usdc));
    }
  }
  if (data.recovery_token) {
    console.log();
    console.log(color("Next:", "bold"));
    console.log(`  export NODEXO_RENTAL_RECOVERY_TOKEN='${data.recovery_token}'`);
    console.log(`  nodexo rental-info ${data.rental_id}`);
    console.log(`  nodexo connect ${data.rental_id}`);
    console.log(`  nodexo rental-extend ${data.rental_id} --hours 4`);
  }
}

function printRentalStatus(data, jsonMode = false) {
  if (jsonMode) {
    console.log(JSON.stringify(data, null, 2));
    return;
  }
  if (data.status === "ended" && data.history) {
    const h = data.history;
    printTitle("Rental ended", "Historical connection details are shown for reference only.");
    printKv("Rental ID", color(h.rental_id, "white", "bold"));
    printKv("Status", h.status || "ended");
    printKv("GPU", `${h.gpu_count || 1}x ${String(h.gpu_model || "GPU").replace(/^NVIDIA /, "")}`);
    if (h.vram_mb) printKv("VRAM", `${Math.round(Number(h.vram_mb) / 1024)} GB`);
    printKv("Started", timeLabel(h.created_at));
    printKv("Ended", timeLabel(h.terminated_at));
    printKv("Duration", secondsLabel(h.actual_seconds || h.ttl_seconds || 0));
    if (h.ssh_host && h.ssh_port) {
      printKv("SSH was", `${h.ssh_user || "root"}@${h.ssh_host}:${h.ssh_port}`);
    }
    return;
  }

  const rental = data.rental || {};
  const conn = rental.connection || {};
  printTitle("Rental active", "Use the recovery token or a signed-in account to manage this rental from another session.");
  printKv("Rental ID", color(rental.rental_id || "--", "white", "bold"));
  printKv("Status", color(rental.status || "running", "green", "bold"));
  if (rental.gpu) {
    printKv("GPU", `${rental.gpu.count || 1}x ${String(rental.gpu.model || "GPU").replace(/^NVIDIA /, "")} · ${rental.gpu.vram_gb || "?"} GB VRAM`);
  }
  if (conn.ssh_host && conn.ssh_port) {
    printKv("SSH", color(sshCommand(conn), "cyan", "bold"));
    printKv("Alias", `nodexo connect ${rental.rental_id}`);
  }
  printKv("Started", timeLabel(rental.created_at));
  if (Number(rental.ttl_seconds || 0) === 0) {
    printKv("Remaining", color("metered", "yellow") + color("  (credit rental; no fixed TTL)", "dim"));
  } else {
    printKv("Remaining", secondsLabel(rental.remaining_seconds));
  }
  if (data.hourly_rate_usdc != null) {
    printKv("Rate", `${money(data.hourly_rate_usdc)}/hr`);
  }
  if (rental.executor_id) printKv("Executor", shortId(rental.executor_id));
  if (rental.container_name) printKv("Container", rental.container_name);

  const ports = data.port_status?.port_status;
  if (ports && ports.total != null) {
    const range = ports.start && ports.end ? `${ports.start}-${ports.end}` : "--";
    printKv("Port pool", `${ports.free ?? "--"}/${ports.total} free · ${range}`);
  }
  console.log();
  console.log(color("Actions:", "bold"));
  console.log(`  nodexo connect ${rental.rental_id}`);
  if (Number(rental.ttl_seconds || 0) > 0) {
    console.log(`  nodexo rental-extend ${rental.rental_id} --hours 4`);
  }
  console.log(`  nodexo rental-key add ${rental.rental_id} --ssh-key ~/.ssh/id_ed25519.pub`);
  console.log(`  nodexo rental-end ${rental.rental_id}`);
}

function printCommandResult(title, data, jsonMode = false) {
  if (jsonMode) {
    console.log(JSON.stringify(data, null, 2));
    return;
  }
  printTitle(title);
  for (const [key, value] of Object.entries(data || {})) {
    if (value == null || typeof value === "object") continue;
    printKv(key.replace(/_/g, " "), String(value));
  }
  if (data?.payment) {
    const payment = data.payment;
    printKv("payment", `${payment.scheme || "x402"} ${payment.status || ""}`.trim());
    if (payment.max_authorized_usdc != null) {
      printKv("max authorized", money(payment.max_authorized_usdc));
    }
  }
}

async function cmdInventory(flags, endpoint = "inventory") {
  const api = apiBase(flags);
  const data = await fetchJson(`${api}/${endpoint}`);
  if (flags.json) {
    console.log(JSON.stringify(data, null, 2));
    return;
  }
  const offers = endpoint === "tiers" ? data.tiers || [] : data.offers || [];
  const rows = offerRows(offers).map((row) => row.slice(1));
  if (!rows.length) {
    console.log("No inventory returned.");
    return;
  }
  const metrics = data.metrics || {};
  printTitle("Nodexo inventory", `${api} - renter-facing capacity`);
  const available = metrics.available_gpus ?? offers.reduce((sum, offer) => sum + Number(offer.available || 0) * Number(offer.gpu_count || 1), 0);
  const total = metrics.total_gpus ?? offers.reduce((sum, offer) => sum + Number(offer.total_gpus || offer.gpu_count || 1), 0);
  console.log(`  ${color("Available", "dim")} ${color(String(available), available ? "green" : "yellow", "bold")} / ${total || offers.length} GPUs`
    + `    ${color("Rented", "dim")} ${metrics.rented_gpus ?? offers.reduce((sum, offer) => sum + Number(offer.rented || 0) * Number(offer.gpu_count || 1), 0)}`
    + `    ${color("Audit", "dim")} ${metrics.audit_gpus ?? offers.reduce((sum, offer) => sum + Number(offer.audit || 0) * Number(offer.gpu_count || 1), 0)}`
    + `    ${color("Classes", "dim")} ${metrics.classes ?? offers.length}`
    + `    ${color("Reliability", "dim")} ${percent(metrics.mean_reliability_24h)}`);
  console.log();
  printTable(rows, ["Inventory", "VRAM", "Avail", "State", "Verify", "Reliability", "Price"]);
  console.log();
  const first = offers.find((offer) => offer.rentable && Number(offer.available || 0) > 0);
  if (first) {
    const short = String(first.gpu_model || "").replace(/^NVIDIA /, "").split(/\s+/).at(-1);
    console.log(`  ${color("Quote", "dim")} nodexo quote --gpu ${short} --duration 1h --storage-gb ${DEFAULT_STORAGE_GB}`);
    console.log(`  ${color("Rent", "dim")}  nodexo rent  --gpu ${short} --duration 1h --storage-gb ${DEFAULT_STORAGE_GB} --ssh-key ~/.ssh/id_ed25519.pub`);
    console.log(`  ${color("Credit", "dim")} nodexo rent --payment credits --gpu ${short} --storage-gb ${DEFAULT_STORAGE_GB} --ssh-key ~/.ssh/id_ed25519.pub`);
  } else {
    console.log(`  ${color("No rentable inventory right now.", "yellow")} Listed rows stay visible so you can see demand and audit state.`);
  }
}

async function cmdMarketplace(flags) {
  if (flags.json || !process.stdin.isTTY) {
    return cmdInventory(flags, "inventory");
  }
  const api = apiBase(flags);
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  try {
    for (;;) {
      const offers = await fetchOffers(api);
      clearScreen();
      printTitle("Nodexo marketplace", "Interactive renter console");
      if (offers.length) {
        printTable(offerRows(offers), ["#", "Inventory", "VRAM", "Avail", "State", "Verify", "Reliability", "Price"]);
      } else {
        console.log("No inventory returned.");
      }
      console.log();
      console.log(`  ${color("[p]", "cyan")} preview quote    ${color("[r]", "cyan")} rent    ${color("[Enter]", "cyan")} refresh    ${color("[q]", "cyan")} quit`);
      const choice = (await rl.question("> ")).trim();
      if (choice.toLowerCase() === "q") return;
      if (choice === "" || choice.toLowerCase() === "refresh") continue;
      if (!["p", "r"].includes(choice.toLowerCase())) continue;

      const rentable = offers.filter((offer) => offer.rentable && Number(offer.available || 0) > 0);
      if (!rentable.length) {
        await rl.question("No rentable inventory right now. Press enter to refresh.");
        continue;
      }
      const defaultIndex = offers.findIndex((offer) => offer.rentable && Number(offer.available || 0) > 0);
      const selectedRaw = await rl.question(`Inventory # [${defaultIndex}]: `);
      const selected = selectedRaw.trim() === "" ? defaultIndex : Number(selectedRaw);
      const offer = offers[selected];
      if (!offer || !offer.rentable || Number(offer.available || 0) <= 0) {
        await rl.question("That inventory row is not rentable. Press enter to continue.");
        continue;
      }
      const hoursRaw = await rl.question("Duration hours [1]: ");
      const hours = hoursRaw.trim() === "" ? 1 : Number(hoursRaw);
      const image = (await rl.question("Cached image [ubuntu:22.04]: ")).trim() || "ubuntu:22.04";
      const sshKey = (await rl.question("SSH public key path [~/.ssh/id_ed25519.pub]: ")).trim()
        || "~/.ssh/id_ed25519.pub";
      const rentFlags = {
        ...flags,
        gpu: offer.gpu_model,
        "gpu-count": String(offer.gpu_count || 1),
        hours: String(hours),
        image,
        "ssh-key": sshKey,
      };
      try {
        if (choice.toLowerCase() === "p") await cmdQuote(rentFlags);
        else await cmdRent(rentFlags);
      } catch (error) {
        console.error(`error: ${error.message}`);
      }
      await rl.question("Press enter to continue.");
    }
  } finally {
    rl.close();
  }
}

async function requestQuote(api, body) {
  const response = await fetch(`${api}/rent`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await readJson(response);
  if (response.status !== 402) {
    const err = new Error(`expected x402 quote, got HTTP ${response.status}: ${JSON.stringify(data).slice(0, 300)}`);
    err.status = response.status;
    err.body = data;
    throw err;
  }
  return data;
}

function rentBodyFromOffer(offer, flags, hours, mode = "x402") {
  const storageGb = numberFlag(flags, "storage-gb", flags["min-storage-gb"] ?? DEFAULT_STORAGE_GB);
  const memoryGb = numberFlag(flags, "memory-gb", flags["min-ram-gb"] ?? 16);
  const body = {
    payment_mode: mode,
    gpu_model: offer.gpu_model,
    gpu_count: Number(offer.gpu_count || 1),
    image: String(flags.image || "ubuntu:22.04"),
    require_cached_image: true,
    pull_image: false,
    ssh_pub_key: readSshPubKey(flags),
    storage_gb: storageGb,
    min_storage_gb: storageGb,
    memory_gb: memoryGb,
    min_ram_gb: memoryGb,
  };
  if (mode === "x402") body.hours = hours;
  if (flags["idempotency-key"]) body.idempotency_key = String(flags["idempotency-key"]);
  return body;
}

function printQuote({ api, offer, hours, quote, body = {}, jsonMode = false }) {
  if (jsonMode) {
    console.log(JSON.stringify({
      offer,
      hours,
      resources: {
        storage_gb: body.storage_gb,
        memory_gb: body.memory_gb,
      },
      quote,
    }, null, 2));
    return;
  }
  const req = quote.accepts?.[0] || {};
  const amount = Number(req.amount || 0) / 1_000_000;
  printTitle("Payment quote", "No payment was signed and no rental was created.");
  printKv("Inventory", `${offer.gpu_count || 1}x ${String(offer.gpu_model || "").replace(/^NVIDIA /, "")}`);
  printKv("Duration", hoursLabel(hours));
  printKv("Resources", `${body.storage_gb || DEFAULT_STORAGE_GB} GB storage · ${body.memory_gb || 16} GB RAM`);
  printKv("Maximum", color(money(amount), "white", "bold"));
  printKv("Scheme", req.scheme || "x402");
  printKv("Network", req.network || "--");
  printKv("Asset", shortAddress(req.asset));
  printKv("Pay to", shortAddress(req.payTo));
  printKv("API", api);
  console.log();
  console.log(color("To rent:", "bold"));
  const short = String(offer.gpu_model || "").replace(/^NVIDIA /, "").split(/\s+/).at(-1);
  console.log(`  export ${DEFAULT_KEY_ENV}=0x...`);
  console.log(`  nodexo rent --gpu ${short} --duration ${hoursLabel(hours)} --storage-gb ${body.storage_gb || DEFAULT_STORAGE_GB} --ssh-key ~/.ssh/id_ed25519.pub`);
}

async function cmdQuote(flags) {
  const api = apiBase(flags);
  const hours = durationHours(flags);
  if (!Number.isInteger(hours) || hours < 1 || hours > 168) {
    throw new Error("--duration must be an integer duration from 1h to 168h for x402 rentals");
  }
  const offer = await selectOffer(api, flags);
  const body = rentBodyFromOffer(offer, flags, hours, "x402");
  const quote = await requestQuote(api, body);
  printQuote({ api, offer, hours, body, quote, jsonMode: Boolean(flags.json) });
}

async function cmdRent(flags) {
  if (flags["dry-run"]) {
    throw new Error("dry-run was replaced by the quote command; run `nodexo quote ...` to preview payment requirements");
  }
  const api = apiBase(flags);
  const mode = paymentMode(flags);
  const offer = await selectOffer(api, flags);
  if (mode === "credits") {
    const body = rentBodyFromOffer(offer, flags, null, "credits");
    const data = await fetchJson(`${api}/rent`, {
      method: "POST",
      headers: {
        ...accountHeaders(flags),
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
    });
    if (flags.save) {
      const file = expandHome(String(flags.save));
      fs.writeFileSync(file, `${JSON.stringify(data, null, 2)}\n`, { mode: 0o600 });
    }
    printRental(data, Boolean(flags.json));
    return;
  }
  const hours = durationHours(flags);
  if (!Number.isInteger(hours) || hours < 1 || hours > 168) {
    throw new Error("--duration must be an integer duration from 1h to 168h for x402 rentals");
  }
  const body = rentBodyFromOffer(offer, flags, hours, "x402");
  const maxUsdc = Number(flags["max-usdc"] || (Number(offer.price_usdc_per_hour || 0) * hours));
  if (!Number.isFinite(maxUsdc) || maxUsdc <= 0) {
    throw new Error("could not infer payment cap; pass --max-usdc");
  }
  const paidFetch = await createPaidFetch(privateKeyFromFlags(flags), usdcToUnitsCeil(maxUsdc));
  const response = await paidFetch(`${api}/rent`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await readJson(response);
  if (!response.ok) throw new Error(`${response.status}: ${JSON.stringify(data).slice(0, 500)}`);
  if (flags.save) {
    const file = expandHome(String(flags.save));
    fs.writeFileSync(file, `${JSON.stringify(data, null, 2)}\n`, { mode: 0o600 });
  }
  printRental(data, Boolean(flags.json));
}

function unitsToUsdc(value) {
  try {
    return Number(BigInt(String(value || "0"))) / 1_000_000;
  } catch {
    return 0;
  }
}

async function cmdCredits(flags) {
  const api = apiBase(flags);
  const data = await fetchJson(`${api}/credits`, {
    headers: accountHeaders(flags),
  });
  if (flags.json) {
    console.log(JSON.stringify(data, null, 2));
    return;
  }
  const balance = data.balance || {};
  printTitle("Credit balance", "Account balance for metered rentals.");
  printKv("Available", color(money(unitsToUsdc(balance.available_units)), "green", "bold"));
  printKv("Reserved", money(unitsToUsdc(balance.reserved_units)));
  printKv("Spent", money(unitsToUsdc(balance.spent_units)));
  const active = Array.isArray(data.segments)
    ? data.segments.filter((segment) => segment.status === "active").length
    : 0;
  printKv("Active rentals", String(active));
}

async function cmdAccountRentals(flags) {
  const api = apiBase(flags);
  const data = await fetchJson(`${api}/account/rentals`, {
    headers: accountHeaders(flags),
  });
  if (flags.json) {
    console.log(JSON.stringify(data, null, 2));
    return;
  }
  printTitle("Account rentals", "Rentals attached to this account.");
  printKv("Open", String(data.active_rental_ids?.length || 0));
  printKv("Recent ended", String(data.recent_ended_rental_ids?.length || 0));
  printKv("Archived", String(data.archived_rental_ids?.length || 0));
  const rows = [
    ...(data.active_rental_ids || []).map((id) => [color("active", "green"), shortId(id)]),
    ...(data.recent_ended_rental_ids || []).map((id) => [color("ended", "dim"), shortId(id)]),
  ];
  if (rows.length) {
    console.log();
    printTable(rows, ["State", "Rental"]);
  }
}

async function cmdAccountSshKeys(flags) {
  const api = apiBase(flags);
  const data = await fetchJson(`${api}/account/ssh-keys`, {
    headers: accountHeaders(flags),
  });
  if (flags.json) {
    console.log(JSON.stringify(data, null, 2));
    return;
  }
  const keys = Array.isArray(data.keys) ? data.keys : [];
  printTitle("Saved SSH keys", "Account keys available for credit-backed automation.");
  if (!keys.length) {
    console.log("  No saved SSH keys.");
    return;
  }
  const rows = keys.map((key) => [
    shortId(key.key_id),
    key.label || "--",
    key.key_type || "--",
    key.fingerprint || "--",
  ]);
  printTable(rows, ["Key", "Label", "Type", "Fingerprint"]);
}

async function cmdAccountSshKeyAdd(flags) {
  const api = apiBase(flags);
  const data = await fetchJson(`${api}/account/ssh-keys`, {
    method: "POST",
    headers: {
      ...accountHeaders(flags),
      "content-type": "application/json",
    },
    body: JSON.stringify({
      public_key: readSshPubKey(flags),
      label: String(flags.label || "").trim() || undefined,
    }),
  });
  if (flags.json) {
    console.log(JSON.stringify(data, null, 2));
    return;
  }
  printCommandResult("Saved SSH key", data.key || data, false);
}

async function cmdAccountSshKeyRemove(flags) {
  const api = apiBase(flags);
  const keyId = mustString(flags, "key-id");
  const data = await fetchJson(`${api}/account/ssh-keys/${encodeURIComponent(keyId)}`, {
    method: "DELETE",
    headers: accountHeaders(flags),
  });
  printCommandResult("Removed SSH key", data, Boolean(flags.json));
}

async function cmdGet(flags) {
  const api = apiBase(flags);
  const rentalId = mustString(flags, "rental-id");
  const data = await fetchJson(`${api}/rentals/${encodeURIComponent(rentalId)}`, {
    headers: authHeaders(flags),
  });
  printRentalStatus(data, Boolean(flags.json));
}

async function cmdEnd(flags) {
  const api = apiBase(flags);
  const rentalId = mustString(flags, "rental-id");
  const data = await fetchJson(`${api}/rentals/${encodeURIComponent(rentalId)}`, {
    method: "DELETE",
    headers: authHeaders(flags),
  });
  printCommandResult("Rental ended", data, Boolean(flags.json));
}

async function cmdExtend(flags) {
  const api = apiBase(flags);
  const rentalId = mustString(flags, "rental-id");
  const hours = numberFlag(flags, "hours");
  if (!Number.isInteger(hours) || hours < 1 || hours > 168) {
    throw new Error("--hours must be an integer from 1 to 168");
  }
  const status = await fetchJson(`${api}/rentals/${encodeURIComponent(rentalId)}`, {
    headers: authHeaders(flags),
  });
  const hourly = Number(status.hourly_rate_usdc || 0);
  const maxUsdc = Number(flags["max-usdc"] || hourly * hours);
  if (!Number.isFinite(maxUsdc) || maxUsdc <= 0) {
    throw new Error("could not infer extension payment cap; pass --max-usdc");
  }
  const paidFetch = await createPaidFetch(privateKeyFromFlags(flags), usdcToUnitsCeil(maxUsdc));
  const response = await paidFetch(`${api}/rentals/${encodeURIComponent(rentalId)}/extend`, {
    method: "POST",
    headers: {
      ...authHeaders(flags),
      "content-type": "application/json",
    },
    body: JSON.stringify({ payment_mode: "x402", hours }),
  });
  const data = await readJson(response);
  if (!response.ok) throw new Error(`${response.status}: ${JSON.stringify(data).slice(0, 500)}`);
  printCommandResult("Rental extended", data, Boolean(flags.json));
}

async function cmdSshKey(flags, method) {
  const api = apiBase(flags);
  const rentalId = mustString(flags, "rental-id");
  const data = await fetchJson(`${api}/rentals/${encodeURIComponent(rentalId)}/ssh-keys`, {
    method,
    headers: {
      ...authHeaders(flags),
      "content-type": "application/json",
    },
    body: JSON.stringify({ ssh_pub_key: readSshPubKey(flags) }),
  });
  printCommandResult(method === "POST" ? "SSH key added" : "SSH key removed", data, Boolean(flags.json));
}

async function main() {
  const [command, ...argv] = process.argv.slice(2);
  if (!command || command === "help" || command === "--help" || command === "-h") usage(0);
  const { flags } = parseArgs(argv);
  try {
    if (command === "inventory") return await cmdInventory(flags, "inventory");
    if (command === "marketplace") return await cmdMarketplace(flags);
    if (command === "tiers") return await cmdInventory(flags, "tiers");
    if (command === "quote") return await cmdQuote(flags);
    if (command === "rent") return await cmdRent(flags);
    if (command === "credits") return await cmdCredits(flags);
    if (command === "account-rentals") return await cmdAccountRentals(flags);
    if (command === "account-ssh-keys") return await cmdAccountSshKeys(flags);
    if (command === "account-ssh-key-add") return await cmdAccountSshKeyAdd(flags);
    if (command === "account-ssh-key-remove") return await cmdAccountSshKeyRemove(flags);
    if (command === "get") return await cmdGet(flags);
    if (command === "end") return await cmdEnd(flags);
    if (command === "extend") return await cmdExtend(flags);
    if (command === "ssh-key-add") return await cmdSshKey(flags, "POST");
    if (command === "ssh-key-remove") return await cmdSshKey(flags, "DELETE");
    usage(1);
  } catch (error) {
    console.error(`error: ${error.message}`);
    if (error.inventory) {
      const rows = error.inventory.map((offer) => [
        `${offer.gpu_count || 1}x ${String(offer.gpu_model || "").replace(/^NVIDIA /, "")}`,
        String(offer.available ?? 0),
        offer.state || "unavailable",
        offer.price_usdc_per_hour == null ? "--" : `${money(offer.price_usdc_per_hour)}/hr`,
      ]);
      if (rows.length) {
        console.error("");
        console.error("Current inventory:");
        printTable(rows, ["Inventory", "Avail", "State", "Price"], (line) => console.error(line));
      }
      console.error("");
      console.error("Run `nodexo inventory` to inspect current capacity, or try again when the listed state is available.");
    }
    process.exit(1);
  }
}

await main();
