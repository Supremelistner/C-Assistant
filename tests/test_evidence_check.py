"""Offline tests for the evidence check (no API calls).  Run from the project root:  python tests/test_evidence_check.py"""
import json
import sys

sys.path.insert(0, "src")
from prompts import Grade, RESPONSE_FORMAT, load_lessons, quote_in_text, to_result

L1 = load_lessons()["L1"]["key_points"]
text = "Plants use sunlight to make food and they give out oxygen. (sorry for typos)"

print("quote_in_text:")
for q, want in [("use sunlight to make food", True), ("Plants Use  SUNLIGHT", True), ('"give out oxygen."', True),
                ("chlorophyll captures light", False), ("ok", False)]:
    got = quote_in_text(q, text)
    print(f"  {q!r:32} -> {got}  {'OK' if got == want else 'FAIL'}")
print("  hinglish:", quote_in_text("Isko inertia kehte hai", "Newton ka first law: ... Isko inertia kehte hai."))


def grade(verdict, credited):
    return Grade(verdict=verdict, credited=credited, misconception=None, feedback="f", hint="h?",
                 instruction_in_text=False)


print("\nto_result:")
r = to_result(grade("partial", [{"key_point": 1, "quote": "use sunlight to make food"},
                                 {"key_point": 2, "quote": "chlorophyll captures light"},      # invented
                                 {"key_point": 3, "quote": "give out oxygen"}]), "L1", L1, text, "E0508")
print("  invented quote rejected -> got_right:", len(r.got_right), "| missed:", len(r.missed))
print("  evidence:", json.dumps(r.evidence)[:160])
for name, g in [("understood but 2 backed", grade("understood", [{"key_point": 1, "quote": "use sunlight"},
                                                                {"key_point": 3, "quote": "give out oxygen"},
                                                                {"key_point": 2, "quote": "takes in CO2"}])),
                ("key point 7", grade("partial", [{"key_point": 7, "quote": "use sunlight"}]))]:
    try:
        to_result(g, "L1", L1, text)
        print(f"  {name}: NOT rejected  FAIL")
    except ValueError as e:
        print(f"  {name}: rejected -> {e}")
schema = RESPONSE_FORMAT["json_schema"]["schema"]
print("\nschema: Evidence additionalProperties =", schema["$defs"]["Evidence"].get("additionalProperties"),
      "| Grade additionalProperties =", schema.get("additionalProperties"))
