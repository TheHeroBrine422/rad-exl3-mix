// extract_pi_tools.mjs OUT.json -- the pi coding agent's built-in tool definitions (name, description, JSON-schema parameters)
// from the installed MIT package @earendil-works/pi-coding-agent (version recorded in the output).
import { createRequire } from "module";
import fs from "fs";
const root = "/usr/local/lib/node_modules/@earendil-works/pi-coding-agent";
const { createAllToolDefinitions } = await import(root + "/dist/core/tools/index.js");
const pkg = JSON.parse(fs.readFileSync(root + "/package.json", "utf8"));
const defs = createAllToolDefinitions("/work");
const out = { package: pkg.name, version: pkg.version, license: pkg.license, tools: {} };
for (const [name, d] of Object.entries(defs))
  out.tools[name] = { name: d.name, description: d.description, parameters: JSON.parse(JSON.stringify(d.parameters)) };
fs.writeFileSync(process.argv[2], JSON.stringify(out, null, 1));
console.log(Object.keys(out.tools), pkg.version);
