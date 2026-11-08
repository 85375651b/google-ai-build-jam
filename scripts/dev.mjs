import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";

const rootDir = process.cwd();
const venvUvicorn = path.join(rootDir, ".venv", "bin", "uvicorn");
const backendCmd = existsSync(venvUvicorn) ? venvUvicorn : "uvicorn";

console.log("Starting Worldbuilding Agent Backend (FastAPI on :8000) & Frontend (Vite on :5173)...");

const backend = spawn(
  backendCmd,
  ["backend.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"],
  { stdio: "inherit", cwd: rootDir }
);

const frontend = spawn(
  "npx",
  ["vite", "--host", "127.0.0.1", "--port", "5173"],
  { stdio: "inherit", cwd: rootDir }
);

function shutdown() {
  backend.kill("SIGTERM");
  frontend.kill("SIGTERM");
  process.exit(0);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
