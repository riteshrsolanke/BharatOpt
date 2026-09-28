# -*- coding: utf-8 -*-
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('services/api'))

from services.api.main import smart_convert_and_solve, SmartSolveRequest, get_historical_trends, sahayak_ai_query, AIQueryRequest

def run_tests():
    print("==================================================")
    print("  RUNNING COMPREHENSIVE BHARATOPT-X TEST SUITE   ")
    print("==================================================")

    # TEST 1: MRPL Crude Oil Blending Template
    print("\n[TEST 1] MRPL Crude Oil Blending Template (LP Simplex)")
    t1 = """We want to maximize refinery gross profit by blending Arab Light, Brent Blend, and Maya Heavy.
Arab Light gives ₹24.50 profit per barrel, Brent gives ₹31.20 profit, and Maya Heavy gives ₹18.75 profit.
Total crude distillation capacity is at most 150,000 barrels.
Desulfurization capacity is limited: Arab takes 1.8 units, Brent takes 0.9 units, and Maya takes 3.4 units, with a maximum of 220,000 units.
We must produce at least 35,000 barrels of high-octane gasoline."""

    req1 = SmartSolveRequest(text=t1, solve_immediately=True)
    res1 = smart_convert_and_solve(req1)
    sol1 = res1["solution_data"]
    print(f"  Status: {sol1['native_status']}")
    print(f"  Objective: INR {sol1['objective']:,.2f}")
    print(f"  Variables: {sol1['vars']}")
    print(f"  Backend: {sol1['backend']}")
    assert sol1["native_status"] == "OPTIMAL", f"Expected OPTIMAL, got {sol1['native_status']}"
    assert sol1["objective"] == 4680000.0, f"Expected 4680000.0, got {sol1['objective']}"
    assert sol1["vars"].get("Brent_Blend") == 150000.0, f"Expected 150000 Brent, got {sol1['vars']}"
    assert "**" not in res1["human_explanation"], "Explanation contains raw markdown asterisks"
    assert "_" not in res1["math_summary"]["objective_formula"], "Formula contains underscores"
    print("  => PASSED (100% Optimal, Accurate to 4.68M INR, Zero Symbols)")

    # TEST 2: Hardware Production Assembly Template (MILP Branch & Bound)
    print("\n[TEST 2] Hardware Production Assembly (MILP Branch & Bound)")
    t2 = """We want to maximize revenue by manufacturing Smartphones, Laptops, and Servers.
Each Smartphone earns ₹450, Laptop earns ₹1800, and Server earns ₹4200.
We have skilled labor limit of at most 1800 hours: Smartphone requires 2.5 hours, Laptop requires 8 hours, and Server requires 18 hours.
Silicon chip stock is at most 2200 units: Smartphone uses 1 chip, Laptop uses 2 chips, and Server uses 8 chips.
All items must be produced in whole integer lot units."""

    req2 = SmartSolveRequest(text=t2, solve_immediately=True)
    res2 = smart_convert_and_solve(req2)
    sol2 = res2["solution_data"]
    print(f"  Status: {sol2['native_status']}")
    print(f"  Objective: INR {sol2['objective']:,.2f}")
    print(f"  Variables: {sol2['vars']}")
    print(f"  Backend: {sol2['backend']}")
    assert sol2["native_status"] == "OPTIMAL", f"Expected OPTIMAL, got {sol2['native_status']}"
    assert sol2["objective"] == 420000.0, f"Expected 420000.0, got {sol2['objective']}"
    assert sol2["vars"].get("Server") == 100.0, f"Expected 100 Servers, got {sol2['vars']}"
    print("  => PASSED (100% Integer Feasible, 420,000 INR revenue)")

    # TEST 3: Clean Energy Portfolio (Convex QP PDHG)
    print("\n[TEST 3] Clean Energy Portfolio (Convex QP PDHG)")
    t3 = """We want to maximize portfolio returns while minimizing quadratic variance risk across Solar, Wind, and Hydro.
Expected returns: Solar gives 12%, Wind gives 14%, and Hydro gives 9%.
Budget limit: total allocation is at most 100%.
Minimize portfolio variance risk penalty."""

    req3 = SmartSolveRequest(text=t3, solve_immediately=True)
    res3 = smart_convert_and_solve(req3)
    sol3 = res3["solution_data"]
    print(f"  Status: {sol3['native_status']}")
    print(f"  Objective: INR {sol3['objective']:,.2f}")
    print(f"  Variables: {sol3['vars']}")
    print(f"  Backend: {sol3['backend']}")
    assert sol3["native_status"] == "OPTIMAL", f"Expected OPTIMAL, got {sol3['native_status']}"
    assert sol3["objective"] > 0, "Expected positive return"
    print("  => PASSED (Convex QP Converged to Global Optimum via PDHG)")

    # TEST 4: Custom Carpentry User Input
    print("\n[TEST 4] Custom User Problem (Carpentry LP)")
    t4 = """We want to maximize profit by making wooden Tables and Chairs.
Tables give 50 profit and Chairs give 30 profit.
We have 100 hours of carpentry. Tables take 4 hours and Chairs take 2 hours.
We have 80 units of wood. Tables take 3 units and Chairs take 2 units."""

    req4 = SmartSolveRequest(text=t4, solve_immediately=True)
    res4 = smart_convert_and_solve(req4)
    sol4 = res4["solution_data"]
    print(f"  Status: {sol4['native_status']}")
    print(f"  Objective: INR {sol4['objective']:,.2f}")
    print(f"  Variables: {sol4['vars']}")
    print(f"  Backend: {sol4['backend']}")
    assert sol4["native_status"] == "OPTIMAL", f"Expected OPTIMAL, got {sol4['native_status']}"
    assert sol4["objective"] == 1300.0, f"Expected 1300.0, got {sol4['objective']}"
    assert sol4["vars"].get("Tables") == 20.0, f"Expected 20 Tables, got {sol4['vars']}"
    assert sol4["vars"].get("Chairs") == 10.0, f"Expected 10 Chairs, got {sol4['vars']}"
    print("  => PASSED (Calculated directly: 20 Tables, 10 Chairs = 1300 INR)")

    # TEST 5: Pure Minimization User Input
    print("\n[TEST 5] Custom User Problem (Power Cost Minimization LP)")
    t5 = """We want to minimize cost of power generation across Coal and Solar.
Coal costs 4 per kWh and Solar costs 2 per kWh.
Total power demand is at least 1000 kWh.
Coal capacity is at most 800 kWh.
Solar capacity is at most 600 kWh."""

    req5 = SmartSolveRequest(text=t5, solve_immediately=True)
    res5 = smart_convert_and_solve(req5)
    sol5 = res5["solution_data"]
    print(f"  Status: {sol5['native_status']}")
    print(f"  Objective: INR {sol5['objective']:,.2f}")
    print(f"  Variables: {sol5['vars']}")
    print(f"  Backend: {sol5['backend']}")
    assert sol5["native_status"] == "OPTIMAL", f"Expected OPTIMAL, got {sol5['native_status']}"
    # Solar capacity is 600 kWh * 2 = 1200, remaining 400 kWh from Coal * 4 = 1600. Total = 2800.
    assert sol5["objective"] == 2800.0, f"Expected 2800.0, got {sol5['objective']}"
    print("  => PASSED (Calculated directly: 600 Solar + 400 Coal = 2800 INR cost)")

    # TEST 6: Historical Trends & Warm-Start API
    print("\n[TEST 6] Historical Trends & Warm-Start API Text Cleanliness")
    trends = get_historical_trends()
    wf = trends["warm_start_forecast"]
    print("  Insight sample:\n", wf["actionable_insight"])
    assert "**" not in wf["actionable_insight"], "Found asterisks in trends insight"
    assert "_" not in wf["actionable_insight"], "Found underscores in trends insight"
    print("  => PASSED (Zero markdown asterisks or underscores)")

    # TEST 7: Interactive What-If Query Cleanliness
    print("\n[TEST 7] Sahayak AI What-If Query Text Cleanliness")
    ai_req = AIQueryRequest(query="Can we expand capacity by 20%?", model_name="MRPL_Crude_Oil_Blending_LP", objective=4680000.0, bottlenecks=["Distillation_Capacity"])
    ai_res = sahayak_ai_query(ai_req)
    print("  Response sample:\n", ai_res["response"].replace('₹', 'Rs.'))
    assert "**" not in ai_res["response"], "Found asterisks in AI response"
    assert "_" not in ai_res["title"], "Found underscores in AI title"
    print("  => PASSED (Clean simple English without symbols)")

    # TEST 8: New Model Template & User Custom Data Accuracy (LP, JSON & Smart)
    print("\n[TEST 8] New Model Creation & Custom User Data Accuracy")
    t8 = """We want to maximize profit from Chairs and Tables.
Chairs give ₹45 profit and Tables give ₹80 profit.
Labor limit: Chairs require 1 hour and Tables require 2 hours, with at most 100 hours available.
Wood limit: Chairs require 2 units and Tables require 3 units, with at most 160 units available."""
    req8 = SmartSolveRequest(text=t8, solve_immediately=True)
    res8 = smart_convert_and_solve(req8)
    sol8 = res8["solution_data"]
    print(f"  Status: {sol8['native_status']}")
    print(f"  Objective: INR {sol8['objective']:,.2f}")
    print(f"  Variables: {sol8['vars']}")
    assert sol8["native_status"] == "OPTIMAL", f"Expected OPTIMAL, got {sol8['native_status']}"
    assert sol8["objective"] == 4100.0, f"Expected 4100.0, got {sol8['objective']}"
    assert sol8["vars"].get("Chairs") == 20.0, f"Expected 20 Chairs, got {sol8['vars'].get('Chairs')}"
    assert sol8["vars"].get("Tables") == 40.0, f"Expected 40 Tables, got {sol8['vars'].get('Tables')}"
    print("  => PASSED (New Model verified: 20 Chairs + 40 Tables = 4,100 INR optimal profit)")

    print("\n==================================================")
    print("  ALL 8 TESTS PASSED SUCCESSFULLY! 100% READY!   ")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
