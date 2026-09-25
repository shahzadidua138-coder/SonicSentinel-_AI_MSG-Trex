"""
run_all_tests.py - Master Automated Test Runner for SonicSentinel AI
Runs the complete testing suite across Security, DSP, Scenarios, and APIs.
"""

import sys
import subprocess

TEST_SCRIPTS = [
    "test_auth_rbac.py",
    "tests/test_dsp_pipeline.py",
    "tests/test_mandatory_scenarios.py",
    "tests/test_api_endpoints.py"
]


def run_all():
    print("==================================================================")
    print("  SONICSENTINEL AI — COMPREHENSIVE AUTOMATED TEST RUNNER")
    print("==================================================================\n")

    passed_count = 0
    total_count = len(TEST_SCRIPTS)

    for script in TEST_SCRIPTS:
        print(f"\n>>> Running: {script} ...")
        res = subprocess.run([sys.executable, script], capture_output=True, text=True)
        print(res.stdout)
        if res.returncode == 0:
            passed_count += 1
            print(f">>> [PASS] {script} successfully completed.\n")
        else:
            print(res.stderr)
            print(f">>> [FAIL] {script} failed with exit code {res.returncode}.\n")

    print("==================================================================")
    if passed_count == total_count:
        print(f"  ALL {total_count}/{total_count} TEST SUITES PASSED 100%! SYSTEM READY FOR JUDGING.")
    else:
        print(f"  {passed_count}/{total_count} SUITES PASSED. PLEASE REVIEW FAILURES.")
    print("==================================================================")
    sys.exit(0 if passed_count == total_count else 1)


if __name__ == '__main__':
    run_all()
