#!/usr/bin/env python3
"""测试 SQL 提取功能"""
import sys
from pathlib import Path

# 添加项目根目录到 Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from bird_dev.bird_text2sql_reward_fn import extract_sql_from_solution

# TEST CASE
test_cases = [
    {
        "name": "Case 1: ### SQL Query: mark",
        "input": """### SQL Query: SELECT * FROM users WHERE id = 1""",
        "expected": "SELECT * FROM users WHERE id = 1"
    },
    {
        "name": "Case 2: ### SQL Query: With residual label",
        "input": """### SQL Query: SELECT * FROM users tool {
}""",
        "expected": "SELECT * FROM users"
    },
    {
        "name": "Case 3: Tool calls + SQL",
        "input": """<tool_call> {"name": "list_tables", "arguments": {}} </tool_call>
{"result": ["users", "orders"]}
SELECT * FROM users""",
        "expected": "SELECT * FROM users"
    },
    {
        "name": "Case 4: Actual log example",
        "input": """


<tool_call> {"name": "list_subject_tree", "arguments": {}} </tool_call>
{"success": 1, "error": null, "result": {"european_football_2": {"player": {"preferred_foot": {"metrics": ["preferred_foot"], "reference_sql": ["preferred_foot"], "knowledge": ["preferred_foot"]}, "attacking_work_rate": {"metrics": ["attacking_work_rate"], "reference_sql": ["attacking_work_rate"], "knowledge": ["attacking_work_rate"]}}}}}
<tool_call> {"name": "get_metrics", "arguments": {"subject_path": ["european_football_2", "player"], "name": "preferred_foot"}} </tool_call>
{"success": 1, "error": null, "result": {"name": "preferred_foot", "description": "The preferred foot of the player", "constraint": "string", "sql_query": "SELECT preferred_foot FROM player"}}
<tool_call> {"name": "get_metrics", "arguments": {"subject_path": ["european_football_2", "player"], "name": "attacking_work_rate"}} </tool_call>
{"success": 1, "error": null, "result": {"name": "attacking_work_rate", "description": "The work rate of the player when attacking", "constraint": "string", "sql_query": "SELECT attacking_work_rate FROM player"}}
<tool_call> {"name": "describe_table", "arguments": {"database": "european_football_2", "table_name": "player"}} </tool_call>
{"success": 1, "error": null, "result": {"columns": [{"name": "player_id", "type": "int", "comment": "Unique identifier for the player", "is_dimension": false}, {"name": "name", "type": "string", "comment": "Name of the player", "is_dimension": false}, {"name": "position", "type": "string", "comment": "Position of the player in the team", "is_dimension": false}, {"name": "preferred_foot", "type": "string", "comment": "The preferred foot of the player", "is_dimension": false}, {"name": "attacking_work_rate", "type": "string", "comment": "The work rate of the player when attacking", "is_dimension": false}]}}
### SQL Query: SELECT COUNT(*) FROM player WHERE preferred_foot = 'left' AND attacking_work_rate = 'low'tool {
}


<tool_call> {"name": "list_subject_tree", "arguments": {}} </tool_call>
{"success": 1, "error": null, "result": {"european_football_2": {"player": {"preferred_foot": {"metrics": ["preferred_foot"], "reference_sql": ["preferred_foot"], "knowledge": ["preferred_foot"]}, "attacking_work_rate": {"metrics": ["attacking_work_rate"], "reference_sql": ["attacking_work_rate"], "knowledge": ["attacking_work_rate"]}}}}}
<tool_call> {"name": "get_metrics", "arguments": {"subject_path": ["european_football_2", "player"], "name": "preferred_foot"}} </tool_call>
{"success": 1, "error": null, "result": {"name": "preferred_foot", "description": "The preferred foot of the player", "constraint": "string", "sql_query": "SELECT preferred_foot FROM player"}}
<tool_call> {"name": "get_metrics", "arguments": {"subject_path": ["european_football_2", "player"], "name": "attacking_work_rate"}} </tool_call>
{"success": 1, "error": null, "result": {"name": "attacking_work_rate", "description": "The work rate of the player when attacking", "constraint": "string", "sql_query": "SELECT attacking_work_rate FROM player"}}
<tool_call> {"name": "describe_table", "arguments": {"database": "european_football_2", "table_name": "player"}} </tool_call>
{"success": 1, "error": null, "result": {"columns": [{"name": "player_id", "type": "int", "comment": "Unique identifier for the player", "is_dimension": false}, {"name": "name", "type": "string", "comment": "Name of the player", "is_dimension": false}, {"name": "position", "type": "string", "comment": "Position of the player in the team", "is_dimension": false}, {"name": "preferred_foot", "type": "string", "comment": "The preferred foot of the player", "is_dimension": false}, {"name": "attacking_work_rate", "type": "string", "comment": "The work rate of the player when attacking", "is_dimension": false}]}}
### SQL Query: SELECT COUNT(*) FROM player WHERE preferred_foot = 'left' AND attacking_work_rate = 'low'tool {
}


<tool_call> {"name": "list_subject_tree", "arguments": {}} </tool_call>
{"success": 1, "error": null, "result": {"european_football_2": {"player": {"preferred_foot": {"metrics": ["preferred_foot"], "reference_sql": ["preferred_foot"], "knowledge": ["preferred_foot"]}, "attacking_work_rate": {"metrics": ["attacking_work_rate"], "reference_sql": ["attacking_work_rate"], "knowledge": ["attacking_work_rate"]}}}}}
<tool_call> {"name": "get_metrics", "arguments": {"subject_path": ["european_football_2", "player"], "name": "preferred_foot"}} </tool_call>
{"success": 1, "error": null, "result": {"name": "preferred_foot", "description": "The preferred foot of the player", "constraint": "string", "sql_query": "SELECT preferred_foot FROM player"}}
<tool_call> {"name": "get_metrics", "arguments": {"subject_path": ["european_football_2", "player"], "name": "attacking_work_rate"}} </tool_call>
{"success": 1, "error": null, "result": {"name": "attacking_work_rate", "description": "The work rate of the player when attacking", "constraint": "string", "sql_query": "SELECT attacking_work_rate FROM player"}}
<tool_call> {"name": "describe_table", "arguments": {"database": "european_football_2", "table_name": "player"}} </tool_call>
{"success": 1, "error": null, "result": {"columns": [{"name": "player_id", "type": "int", "comment": "Unique identifier for the player", "is_dimension": false}, {"name": "name", "type": "string", "comment": "Name of the player", "is_dimension": false}, {"name": "position", "type": "string", "comment": "Position of the player in the team", "is_dimension": false}, {"name": "preferred_foot", "type": "string", "comment": "The preferred foot of the player", "is_dimension": false}, {"name": "attacking_work_rate", "type": "string", "comment": "The work rate of the player when attacking", "is_dimension": false}]}}
### SQL Query: SELECT COUNT(*) FROM player WHERE preferred_foot = 'left' AND attacking_work_rate = 'low'""",
        "expected": "SELECT COUNT(*) FROM player WHERE preferred_foot = 'left' AND attacking_work_rate = 'low'"
    },
    {
        "name": "Case 5: With EOS token",
        "input": """<|im_end|>
SELECT * FROM users WHERE age > 18""",
        "expected": "SELECT * FROM users WHERE age > 18"
    },
    {
        "name": "Case 6: 只有 tool call，没有 SQL",
        "input": """<tool_call> {"name": "list_tables", "arguments": {}} </tool_call>
I cannot answer this question.""",
        "expected": None
    },
    {
        "name": "Case 7: Markdown code block",
        "input": """### SQL Query:
```sql
SELECT * FROM products
```""",
        "expected": "SELECT * FROM products"
    },
    # ==================== Destructive/Edge Case Tests ====================
    {
        "name": "Edge 1: Empty string",
        "input": "",
        "expected": None
    },
    {
        "name": "Edge 2: Only whitespace and newlines",
        "input": "   \n\n\t  \n  ",
        "expected": None
    },
    {
        "name": "Edge 3: Only marker without SQL",
        "input": "### SQL Query:",
        "expected": None
    },
    {
        "name": "Edge 4: Multiple markers (should take last one)",
        "input": """### SQL Query: SELECT * FROM old_table
### SQL Query: SELECT * FROM new_table""",
        "expected": "SELECT * FROM new_table"
    },
    {
        "name": "Edge 5: SQL keyword in JSON (should not be recognized)",
        "input": """<tool_call> {"name": "test", "arguments": {}} </tool_call>
{"message": "SELECT is a keyword"}
No SQL here""",
        "expected": None
    },
    {
        "name": "Edge 6: Unmatched tool tags",
        "input": """<tool_call> {"name": "test"} </tool_call>
<tool_call>
SELECT * FROM users""",
        "expected": "SELECT * FROM users"
    },
    {
        "name": "Edge 7: Nested tool tags",
        "input": """<tool_call>
  <tool_call> {"nested": true} </tool_call>
</tool_call>
SELECT * FROM nested_test""",
        "expected": "SELECT * FROM nested_test"
    },
    {
        "name": "Edge 8: SQL syntax error (misspelled SELECT)",
        "input": """### SQL Query: SELCT * FORM users WERE id = 1""",
        "expected": "SELCT * FORM users WERE id = 1"  # With marker, we extract even invalid SQL
    },
    {
        "name": "Edge 9: Very long garbage followed by SQL",
        "input": "x" * 1000 + "\n### SQL Query: SELECT * FROM long_test",
        "expected": "SELECT * FROM long_test"
    },
    {
        "name": "Edge 10: Unicode and special characters",
        "input": """### SQL Query: SELECT * FROM users WHERE name = 'test' AND emoji = '😀'""",
        "expected": "SELECT * FROM users WHERE name = 'test' AND emoji = '😀'"
    },
    {
        "name": "Edge 11: SQL with comments",
        "input": """### SQL Query:
-- This is a comment
SELECT * FROM users""",
        "expected": "-- This is a comment\nSELECT * FROM users"
    },
    {
        "name": "Edge 12: Multiple EOS tokens",
        "input": """<|im_end|><|im_end|><|endoftext|>
SELECT * FROM multiple_eos""",
        "expected": "SELECT * FROM multiple_eos"
    },
    {
        "name": "Edge 13: WITH CTE (should be recognized)",
        "input": """### SQL Query: WITH cte AS (SELECT 1) SELECT * FROM cte""",
        "expected": "WITH cte AS (SELECT 1) SELECT * FROM cte"
    },
    {
        "name": "Edge 14: Mixed terminators",
        "input": """### SQL Query: SELECT * FROM test</think><think>tool {
garbage here""",
        "expected": "SELECT * FROM test"
    },
    {
        "name": "Edge 15: Only tool call without closing tag",
        "input": """<tool_call> {"name": "test", "arguments": {}}
No closing tag and no SQL""",
        "expected": None
    },
    {
        "name": "Edge 16: SQL in markdown without marker",
        "input": """```sql
SELECT * FROM markdown_only
```""",
        "expected": None  # No SELECT in first token after code block removal
    },
    {
        "name": "Edge 17: Empty tool_call followed by SQL",
        "input": """<tool_call></tool_call>
SELECT * FROM empty_call""",
        "expected": "SELECT * FROM empty_call"
    },
    {
        "name": "Edge 18: Nested markdown blocks",
        "input": """### SQL Query:
```sql
```sql
SELECT * FROM nested
```
```""",
        "expected": "SELECT * FROM nested"
    },
    {
        "name": "Edge 19: Lowercase select (should be recognized)",
        "input": """select * from lowercase""",
        "expected": "select * from lowercase"  # is_sql_solution uses .upper(), accepts lowercase
    },
    {
        "name": "Edge 20: Mixed case SELECT",
        "input": """### SQL Query: SeLeCt * FrOm MiXeD""",
        "expected": "SeLeCt * FrOm MiXeD"
    },
]

def run_tests():
    print("=" * 80)
    print("SQL Extraction Test Suite")
    print("=" * 80)

    passed = 0
    failed = 0

    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}] {test['name']}")
        print(f"Input (first 100 chars): {test['input'][:100]}...")

        result = extract_sql_from_solution(test['input'])
        expected = test['expected']

        if result == expected:
            print(f"✅ PASS")
            print(f"   Output: {result}")
            passed += 1
        else:
            print(f"❌ FAIL")
            print(f"   Expected: {expected}")
            print(f"   Got:      {result}")
            failed += 1

    print("\n" + "=" * 80)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)

if __name__ == "__main__":
    run_tests()
