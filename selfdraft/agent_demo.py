#!/usr/bin/env python3
"""
Minimal ReAct tool-using agent loop, used to show that self-draft cuts END-TO-END
agent latency on Arm64 cloud. Agents emit long, structured, predictable text
(Thought/Action chains) - exactly where MTP self-speculation has high acceptance.

Standalone: point at a running llama.cpp server (--port) and it runs the loop.
Or import run_agent() (sd.py's `agent` subcommand times vanilla vs self-draft).
"""
import ast, json, operator, re, time, urllib.request

# safe arithmetic eval for the CALC tool
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}
def _calc(expr):
    def ev(n):
        if isinstance(n, ast.Constant): return n.value
        if isinstance(n, ast.BinOp):    return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp):  return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsupported")
    return ev(ast.parse(expr.strip(), mode="eval").body)

TASK = ("A warehouse has 3 shelves with 14 boxes each, plus 2 shelves with 9 boxes each. "
        "Each box holds 6 items. Then 17 items are found damaged and removed. "
        "How many usable items remain?")

SYS = ("You are a careful reasoning agent. Solve the task step by step. "
       "For EVERY arithmetic operation you MUST use the tool by writing a line exactly like:\n"
       "Action: CALC[ <expression> ]\n"
       "Do not compute arithmetic yourself. After you see the Observation, continue. "
       "When you have the final number, write:\n"
       "FINAL: <number>\n")

def _prompt(history):
    body = SYS + "\nTask: " + TASK + "\n\n" + history + "Thought:"
    return f"<start_of_turn>user\n{body}<end_of_turn>\n<start_of_turn>model\n"

def run_agent(port, max_steps=10, host="127.0.0.1", verbose=False):
    """Run the ReAct loop against a server. Returns dict with answer, steps, wall_ms, tokens."""
    history, steps, tokens = "", 0, 0
    t0 = time.monotonic()
    answer = None
    for _ in range(max_steps):
        body = json.dumps({"prompt": _prompt(history), "n_predict": 256, "temperature": 0,
                           "top_k": 1, "cache_prompt": False,
                           "stop": ["Observation:", "<end_of_turn>"]}).encode()
        req = urllib.request.Request(f"http://{host}:{port}/completion", body,
                                     {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=600))
        out = r["content"]
        tokens += r["timings"]["predicted_n"]
        steps += 1
        if verbose:
            print(f"--- step {steps} ---\n{out.strip()}")
        # final answer present?
        if "FINAL" in out:
            m = re.search(r"FINAL:\s*([-\d.]+)", out)
            answer = m.group(1) if m else None
            break
        # tool call?
        mcalc = re.findall(r"CALC\[\s*(.+?)\s*\]", out)
        if mcalc:
            try:
                obs = _calc(mcalc[-1])
            except Exception as e:
                obs = f"error: {e}"
            history += "Thought:" + out.rstrip() + f"\nObservation: {obs}\n"
        else:
            history += "Thought:" + out.rstrip() + "\nObservation: (no tool call; continue)\n"
    wall_ms = (time.monotonic() - t0) * 1000.0
    return {"answer": answer, "steps": steps, "wall_ms": wall_ms, "tokens": tokens,
            "correct": (answer is not None and abs(float(answer) - 343) < 1e-6)}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    res = run_agent(a.port, host=a.host, verbose=a.verbose)
    print(json.dumps(res, indent=2))
