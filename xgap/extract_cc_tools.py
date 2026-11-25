#!/usr/bin/env python3
"""extract_cc_tools.py SDK_TOOLS.d.ts OUT.json -- Claude Code's built-in tool INPUT schemas (names, parameters, parameter
descriptions) from @anthropic-ai/claude-agent-sdk's generated `sdk-tools.d.ts`, as JSON-schema function definitions.
The tool-level `description` is NOT in that file (the harness's long tool prompts are not published as data): it is a one-line
summary written here, and says so in the output (`description_source`)."""
import json, re, sys

SUMMARY = {   # tool name -> (interface, one-line summary)
    "Bash": ("BashInput", "Run a shell command in the project's working directory and return its output."),
    "Read": ("FileReadInput", "Read a file from the local filesystem."),
    "Edit": ("FileEditInput", "Replace an exact string in an existing file."),
    "Write": ("FileWriteInput", "Write a file to the local filesystem, overwriting an existing one."),
    "Glob": ("GlobInput", "Find files by glob pattern."),
    "Grep": ("GrepInput", "Search file contents with a regular expression (ripgrep)."),
    "NotebookEdit": ("NotebookEditInput", "Edit a cell of a Jupyter notebook."),
    "TodoWrite": ("TodoWriteInput", "Create and update the structured task list for the session."),
    "WebFetch": ("WebFetchInput", "Fetch a URL and answer a prompt about its content."),
    "WebSearch": ("WebSearchInput", "Search the web."),
    "Agent": ("AgentInput", "Launch a subagent to handle a task."),
    "Task": ("AgentInput", "Launch a subagent to handle a task."),
    "ExitPlanMode": ("ExitPlanModeInput", "Present a plan and leave plan mode."),
    "TaskStop": ("TaskStopInput", "Stop a running background task."),
    "AskUserQuestion": ("AskUserQuestionInput", "Ask the user a multiple-choice question."),
    "TaskCreate": ("TaskCreateInput", "Create a task in the task list."),
    "TaskGet": ("TaskGetInput", "Get a task from the task list."),
    "TaskUpdate": ("TaskUpdateInput", "Update a task in the task list."),
    "TaskList": ("TaskListInput", "List the tasks in the task list."),
    "ScheduleWakeup": ("ScheduleWakeupInput", "Schedule when to resume work in a dynamic loop."),
    "Monitor": ("MonitorInput", "Stream events from a background process."),
    "EnterPlanMode": ("EnterPlanModeInput", "Enter plan mode."),
}


class P:
    def __init__(self, s): self.s, self.i = s, 0

    def ws(self):
        while self.i < len(self.s) and self.s[self.i] in " \t\r\n": self.i += 1

    def doc(self):
        """a /** ... */ comment at the cursor (after whitespace) -> its text, else None; line comments skipped"""
        self.ws(); d = None
        while self.s.startswith("/*", self.i) or self.s.startswith("//", self.i):
            if self.s.startswith("//", self.i):
                self.i = self.s.find("\n", self.i) + 1; self.ws(); continue
            j = self.s.index("*/", self.i)
            d = re.sub(r"^\s*\* ?", "", self.s[self.i + 3:j], flags=re.M).strip()
            self.i = j + 2; self.ws()
        return d

    def peek(self, t): self.ws(); return self.s.startswith(t, self.i)

    def eat(self, t):
        self.ws()
        assert self.s.startswith(t, self.i), (t, self.s[self.i:self.i + 80])
        self.i += len(t)

    def ident_or_str(self):
        self.ws()
        if self.s[self.i] in "\"'":
            q = self.s[self.i]; j = self.s.index(q, self.i + 1); v = self.s[self.i + 1:j]; self.i = j + 1; return v
        m = re.compile(r"[A-Za-z_$][\w$]*").match(self.s, self.i); self.i = m.end(); return m.group(0)

    def object(self):
        self.eat("{"); props, req = {}, []
        while True:
            d = self.doc()
            if self.peek("}"): self.eat("}"); break
            if self.peek("["):   # index signature [k: string]: T
                j = self.s.index("]", self.i); self.i = j + 1; self.eat(":"); t = self.type(); self.peek(";") and self.eat(";")
                continue
            name = self.ident_or_str(); opt = False
            if self.peek("?"): self.eat("?"); opt = True
            self.eat(":"); t = self.type()
            if d: t = dict(t, description=d)
            props[name] = t
            if not opt: req.append(name)
            if self.peek(";"): self.eat(";")
            elif self.peek(","): self.eat(",")
        o = {"type": "object", "properties": props}
        if req: o["required"] = req
        return o

    def primary(self):
        self.ws(); c = self.s[self.i]
        if c == "{": t = self.object()
        elif c == "(": self.eat("("); t = self.type(); self.eat(")")
        elif c == "[":      # tuple (json-schema-to-typescript output for minItems / maxItems): an array of its first element type
            self.eat("["); items = []
            while not self.peek("]"):
                if self.peek("..."): self.eat("...")
                items.append(self.type())
                if self.peek(","): self.eat(",")
            self.eat("]"); t = {"type": "array", "items": items[0] if items else {}}
        elif c in "\"'":
            q = c; j = self.s.index(q, self.i + 1); t = {"const": self.s[self.i + 1:j]}; self.i = j + 1
        else:
            m = re.compile(r"[A-Za-z_$][\w$.]*").match(self.s, self.i); w = m.group(0); self.i = m.end()
            if w in ("string", "number", "boolean"): t = {"type": w}
            elif w in ("unknown", "any"): t = {}
            elif w == "null": t = {"type": "null"}
            elif w == "never": t = {}
            elif w == "Record":
                self.eat("<"); self.type(); self.eat(","); v = self.type(); self.eat(">"); t = {"type": "object", "additionalProperties": v}
            else: t = {}
        while self.peek("[]"): self.eat("[]"); t = {"type": "array", "items": t}
        return t

    def type(self):
        if self.peek("|"): self.eat("|")
        alts = [self.primary()]
        while self.peek("|"): self.eat("|"); alts.append(self.primary())
        alts = [a for a in alts if a != {"type": "null"}] if len(alts) > 1 else alts
        if len(alts) == 1:
            a = alts[0]; return {"type": "string", "enum": [a["const"]]} if "const" in a else a
        if all("const" in a for a in alts): return {"type": "string", "enum": [a["const"] for a in alts]}
        return {"anyOf": alts}


def interface(src, name):
    m = re.search(r"export interface %s \{" % name, src)
    if not m: return None
    p = P(src); p.i = m.end() - 1
    return p.object()


def main():
    src = open(sys.argv[1]).read()
    ver = re.search(r'"version":\s*"([^"]+)"', open(sys.argv[1].replace("sdk-tools.d.ts", "package.json")).read()).group(1)
    out = {"package": "@anthropic-ai/claude-agent-sdk", "version": ver, "file": "sdk-tools.d.ts",
           "description_source": "one-line summary written by xgap (the SDK file carries parameter descriptions only)", "tools": {}}
    for name, (iface, summ) in SUMMARY.items():
        sch = interface(src, iface)
        if sch is None: print("missing", iface); continue
        out["tools"][name] = {"name": name, "description": summ, "parameters": sch}
    json.dump(out, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
    print(ver, {k: len(v["parameters"]["properties"]) for k, v in out["tools"].items()})


if __name__ == "__main__":
    main()
