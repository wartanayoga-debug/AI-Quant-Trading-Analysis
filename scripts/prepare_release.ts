import { cpSync, existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";

const root = process.cwd();
const outDir = join(root, "release");
const staging = join(tmpdir(), "quantprime-v6-source");
const zip = join(outDir, "quantprime-v6-source.zip");
const excluded = [
  "node_modules", "dist", "release", ".git", ".venv", ".venv-ag", "assets", "python_bridge/__pycache__",
  "data/feature_store.db", "data/db.json", "data/user_state.json", "data/alerts.json",
];

function run(command: string, args: string[]) {
  const result = spawnSync(command, args, { cwd: root, shell: true, stdio: "inherit" });
  if (result.status !== 0) throw new Error(`${command} ${args.join(" ")} failed`);
}

function shouldExclude(source: string): boolean {
  const normalizedRoot = root.replace(/^\\\\\?\\/, "").replaceAll("\\", "/").toLowerCase();
  const normalizedSource = source.replace(/^\\\\\?\\/, "").replaceAll("\\", "/").toLowerCase();
  const rel = normalizedSource.startsWith(`${normalizedRoot}/`)
    ? normalizedSource.slice(normalizedRoot.length + 1)
    : normalizedSource;
  return excluded.some((item) => rel === item || rel.startsWith(`${item}/`))
    || /\.(sqlite|sqlite-wal|sqlite-shm|db|log|rar|zip)$/i.test(rel);
}

run("npm", ["run", "lint"]);
run("npm", ["test"]);
run("npm", ["run", "build"]);

mkdirSync(outDir, { recursive: true });
if (existsSync(staging)) rmSync(staging, { recursive: true, force: true });
if (existsSync(zip)) rmSync(zip, { force: true });
cpSync(root, staging, { recursive: true, filter: (source) => !shouldExclude(source) });

const report = `# QuantPrime V6 Release Report

- App version: 6.0.0
- Date: ${new Date().toISOString()}
- Lint: PASS
- Test: PASS
- Build: PASS
- Python bridge: not modified by release script
- Excluded: ${excluded.join(", ")}, runtime databases, logs, and archives
- Known limitation: paper/manual execution only; no broker auto-execution
`;
writeFileSync(join(staging, "RELEASE_REPORT.md"), report, "utf8");
run("powershell", ["-NoProfile", "-Command", `Compress-Archive -Path '${staging}\\*' -DestinationPath '${zip}' -Force`]);
console.log(`Release created: ${zip}`);
