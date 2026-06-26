// PM2 ecosystem config for Nodexo operators.
//
// Copy and fill in your values:
//   cp ecosystem.config.example.cjs ecosystem.config.cjs
//
// Bootstrap the host once before starting PM2:
//   bash scripts/setup_miner.sh
//   bash scripts/setup_validator.sh
//   bash scripts/setup_endpoint_proxy.sh --role miner --public-port 8091
//   bash scripts/setup_endpoint_proxy.sh --role validator --public-port 9443
//
// The proxy defaults are public 8091 -> backend 127.0.0.1:18091 for miners
// and public 9443 -> backend 127.0.0.1:19443 for validators.
//
// Network:
//   --subtensor-network test    # public testnet preview, netuid 468
//   --subtensor-network finney  # mainnet, netuid 106 after mainnet config ships
//   NODEXO_SUBTENSOR_ENDPOINT=ws://127.0.0.1:9944  # optional private RPC
//
// Usage:
//   pm2 start ecosystem.config.cjs --only miner-nodexo
//   pm2 start ecosystem.config.cjs --only vali-nodexo
//   pm2 start ecosystem.config.cjs --only receipt-ingress
//   pm2 logs miner-nodexo --lines 80
//   pm2 restart miner-nodexo --update-env
//   pm2 save

const fs = require("fs");

function readEnvFile(file) {
  const out = {};
  try {
    for (const rawLine of fs.readFileSync(file, "utf8").split(/\r?\n/)) {
      const line = rawLine.trim();
      if (!line || line.startsWith("#")) continue;
      const idx = line.indexOf("=");
      if (idx <= 0) continue;
      const key = line.slice(0, idx).trim();
      let value = line.slice(idx + 1).trim();
      if (
        (value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'"))
      ) {
        value = value.slice(1, -1);
      }
      out[key] = value;
    }
  } catch (_err) {
    // scripts/setup_validator.sh writes this file. Missing is fine while an
    // operator is still editing the example.
  }
  return out;
}

const validatorHostEnv = readEnvFile(
  process.env.NODEXO_VALIDATOR_ENV_FILE || "/etc/nodexo/validator.env",
);
const validatorDataDir =
  process.env.NODEXO_DATA_DIR ||
  validatorHostEnv.NODEXO_DATA_DIR ||
  `${process.env.HOME}/.nodexo`;
const validatorDbUrl =
  process.env.DB_URL ||
  process.env.NODEXO_VALIDATOR_DB_URL ||
  validatorHostEnv.DB_URL ||
  validatorHostEnv.NODEXO_VALIDATOR_DB_URL ||
  "postgresql+psycopg2://nodexo:CHANGE_ME@127.0.0.1:5432/nodexo_validator";

module.exports = {
  apps: [
    {
      name: "miner-nodexo",
      script: ".venv/bin/python",
      args: [
        "-u",
        "-m", "neurons.miner.miner",
        "--wallet", "<WALLET>",
        "--hotkey", "<HOTKEY>",
        "--subtensor-network", "test",
        "--port", "18091",
        "--bind-host", "127.0.0.1",
        "--endpoint", "http://<YOUR_PUBLIC_IP_OR_DOMAIN>:8091",
        "--rental-port-start", "20000",
        "--rental-port-end", "20100",
        "--auto-update",
      ].join(" "),
      cwd: __dirname,
      env: {
        PYTHONPATH: `${__dirname}:${__dirname}/zkgemm/cuda`,
        PYTHONUNBUFFERED: "1",
        NODEXO_DATA_DIR: `${process.env.HOME}/.nodexo`,
        PROOF_EPOCH_BLOCKS: "15",
        NODEXO_SUBNET_CONFIG_URL: "https://validator.nodexo.ai/subnet-config",
        NODEXO_SUBNET_CONFIG_REFRESH_SECONDS: "600",
        NODEXO_SUBTENSOR_ENDPOINT:
          process.env.NODEXO_SUBTENSOR_ENDPOINT || "",
        MINER_RENTAL_PORT_PREFLIGHT: "strict",
        MINER_RENTAL_PORT_PREFLIGHT_SCOPE: "sample",
        MINER_RENTAL_IMAGE_PREFLIGHT: "strict",
        MINER_OPTIONAL_IMAGE_PREFLIGHT: "warn",
        // Rental/container authority. Production miners normally get the
        // primary validator hotkey from subnet runtime config. Keep
        // strict mode enabled; env hotkeys are for local/emergency override.
        NODEXO_STRICT_ALLOWLIST: process.env.NODEXO_STRICT_ALLOWLIST || "1",
        NODEXO_ALLOWED_VALIDATOR_HOTKEYS:
          process.env.NODEXO_ALLOWED_VALIDATOR_HOTKEYS || "",
        NODEXO_VALIDATOR_POST_TIMEOUT_SECONDS: "30",
        NODEXO_VALIDATOR_BROADCAST_DEADLINE_SECONDS: "70",
        // Set to 1 only when the endpoint is a real HTTPS URL with a
        // validator-trusted certificate.
        MINER_STRICT_PUBLIC_ENDPOINT: "0",
        MINER_ALLOW_LOCAL_ENDPOINT: "0",
        ZKGEMM_CUDA_MANIFEST_URL:
          "https://pub-ef00d9a98f734d94af3c8904eba0eb11.r2.dev/zkgemm/v0.1.2/manifest.json",
      },
      autorestart: true,
      max_restarts: 5,
      min_uptime: "60s",
      restart_delay: 10000,
      merge_logs: true,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      max_size: "50M",
      retain: 3,
    },
    {
      name: "vali-nodexo",
      script: ".venv/bin/python",
      args: [
        "-u",
        "-m", "neurons.validator.validator",
        "--wallet", "<WALLET>",
        "--hotkey", "<HOTKEY>",
        "--subtensor-network", "test",
        "--port", "19443",
        "--bind-host", "127.0.0.1",
        "--endpoint", "http://<YOUR_PUBLIC_IP_OR_DOMAIN>:9443",
        "--auto-update",
      ].join(" "),
      cwd: __dirname,
      env: {
        ...validatorHostEnv,
        PYTHONPATH: `${__dirname}:${__dirname}/zkgemm/cuda`,
        PYTHONUNBUFFERED: "1",
        NODEXO_DATA_DIR: validatorDataDir,
        NODEXO_VALIDATOR_DB_URL: validatorDbUrl,
        DB_URL: validatorDbUrl,
        NODEXO_EVM_RPC_URL:
          process.env.NODEXO_EVM_RPC_URL ||
          validatorHostEnv.NODEXO_EVM_RPC_URL ||
          "",
        NODEXO_SUBTENSOR_HTTP_RPC_URL:
          process.env.NODEXO_SUBTENSOR_HTTP_RPC_URL ||
          validatorHostEnv.NODEXO_SUBTENSOR_HTTP_RPC_URL ||
          "",
        NODEXO_SUBTENSOR_HTTP_RPC_URL_TEST:
          process.env.NODEXO_SUBTENSOR_HTTP_RPC_URL_TEST ||
          validatorHostEnv.NODEXO_SUBTENSOR_HTTP_RPC_URL_TEST ||
          "",
        NODEXO_SUBTENSOR_HTTP_RPC_URL_FINNEY:
          process.env.NODEXO_SUBTENSOR_HTTP_RPC_URL_FINNEY ||
          validatorHostEnv.NODEXO_SUBTENSOR_HTTP_RPC_URL_FINNEY ||
          "",
        NODEXO_SUBTENSOR_ENDPOINT:
          process.env.NODEXO_SUBTENSOR_ENDPOINT ||
          validatorHostEnv.NODEXO_SUBTENSOR_ENDPOINT ||
          "",
        VALIDATOR_REQUIRE_POSTGRES:
          process.env.VALIDATOR_REQUIRE_POSTGRES ||
          validatorHostEnv.VALIDATOR_REQUIRE_POSTGRES ||
          "1",
        PROOF_EPOCH_BLOCKS: "15",
        NUM_CHALLENGES: "4",
        VERIFY_QUEUE_SIZE: "2000",
        VALIDATOR_BIND_HOST: "127.0.0.1",
        VALIDATOR_ALLOW_LOCAL_ENDPOINT: "0",
        // Set to 1 only when the endpoint is a real HTTPS URL with a miner-
        // trusted certificate.
        VALIDATOR_STRICT_PUBLIC_ENDPOINT: "0",
        VALIDATOR_DB_PRUNE_ENABLED: "1",
        NODEXO_SUBNET_CONFIG_URL: "https://nodexo.ai/api/subnet-config",
        NODEXO_SUBNET_CONFIG_REFRESH_SECONDS: "600",
        // Only the primary validator runs canaries. Other validators leave
        // this disabled.
        CANARY_ENABLED: "0",
        CANARY_MEAN_SECONDS: "86400",
        // auto = EVM validators publish ValidatorRegistry only; no-EVM
        // validators publish native Bittensor axon only. Miner discovery is
        // permit-gated for both paths. Set "both" only when intentionally
        // testing both discovery paths for one UID.
        NODEXO_VALIDATOR_DISCOVERY_MODE: "auto",
        VALIDATOR_DEACTIVATE_ON_SHUTDOWN: "0",
        // Set to 1 for a read/verify validator that should not perform EVM
        // writes. This disables ValidatorRegistry registration, canaries,
        // rental control, and reportOffline while keeping proof verification
        // and weight setting active.
        VALIDATOR_NO_EVM: "1",
      },
      autorestart: true,
      max_restarts: 5,
      min_uptime: "60s",
      restart_delay: 10000,
      // Allow graceful FastAPI shutdown hooks to finish before PM2 force-kills
      // the process.
      kill_timeout: 60000,
      merge_logs: true,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      max_size: "50M",
      retain: 3,
    },
    {
      name: "receipt-ingress",
      script: ".venv/bin/python",
      args: [
        "-u",
        "-m", "neurons.validator.receipt_ingress",
      ].join(" "),
      cwd: __dirname,
      env: {
        ...validatorHostEnv,
        PYTHONPATH: `${__dirname}:${__dirname}/zkgemm/cuda`,
        PYTHONUNBUFFERED: "1",
        NODEXO_DATA_DIR: validatorDataDir,
        NODEXO_VALIDATOR_DB_URL: validatorDbUrl,
        DB_URL: validatorDbUrl,
        NODEXO_RECEIPT_BIND_HOST: "127.0.0.1",
        NODEXO_RECEIPT_PORT: "19444",
      },
      autorestart: true,
      max_restarts: 10,
      min_uptime: "30s",
      restart_delay: 5000,
      merge_logs: true,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      max_size: "50M",
      retain: 3,
    },
  ],
};
