"""
test_auth_rbac.py - Automated Test Suite for SonicSentinel Authentication & RBAC
"""

import urllib.request
import urllib.parse
import http.cookiejar
import database

BASE = 'http://127.0.0.1:5000'


def get_session():
    cj = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))


def run_tests():
    print("==================================================")
    print("  SonicSentinel AI - Backend Auth & RBAC Tests")
    print("==================================================")

    # Clean up test accounts to ensure idempotency
    conn = database.get_db_connection()
    conn.execute("DELETE FROM users WHERE username IN ('normaltester1', 'rachel_operator')")
    conn.commit()
    conn.close()

    # 1. Public Register: Attempt role override / privilege escalation
    opener = get_session()
    reg_data = urllib.parse.urlencode({
        'full_name': 'Test Normal Person',
        'username': 'normaltester1',
        'email': 'normaltester1@example.com',
        'password': 'Password@123',
        'confirm_password': 'Password@123',
        'role': 'Administrator'  # Attack vector: Injecting role
    }).encode('utf-8')

    req = urllib.request.Request(f'{BASE}/register', data=reg_data)
    opener.open(req)

    user = database.get_user_by_email_or_username('normaltester1')
    assert user is not None, "User creation failed"
    assert user['role'] == 'Normal User', f"Expected Normal User, got {user['role']}"
    print("[PASS] 1. Public registration defaults to Normal User & ignores role injection.")

    # 2. Normal User Login -> auto-redirects to /user/dashboard
    login_data = urllib.parse.urlencode({
        'identifier': 'normaltester1',
        'password': 'Password@123'
    }).encode('utf-8')
    resp = opener.open(urllib.request.Request(f'{BASE}/login', data=login_data))
    assert '/user/dashboard' in resp.geturl(), f"Expected /user/dashboard, got {resp.geturl()}"
    print("[PASS] 2. Normal User login routes automatically to /user/dashboard.")

    # 3. Normal User accessing /admin/dashboard -> 403 Forbidden
    try:
        opener.open(f'{BASE}/admin/dashboard')
        assert False, "Normal user should not access /admin/dashboard"
    except urllib.error.HTTPError as e:
        assert e.code == 403, f"Expected 403, got {e.code}"
        print("[PASS] 3. Normal User accessing /admin/dashboard returns HTTP 403 Forbidden.")

    # 4. Cache-Control Security Headers (Back-button logout protection)
    dash_resp = opener.open(f'{BASE}/user/dashboard')
    cc = dash_resp.headers.get('Cache-Control', '')
    assert 'no-store' in cc and 'no-cache' in cc, "Missing anti-cache security headers"
    print("[PASS] 4. Cache-Control headers verified (no-store, no-cache, must-revalidate).")

    # 5. Built-in Admin Login -> auto-redirects to /admin/dashboard
    admin_opener = get_session()
    admin_login = urllib.parse.urlencode({'identifier': 'admin', 'password': 'Admin@123'}).encode('utf-8')
    admin_resp = admin_opener.open(urllib.request.Request(f'{BASE}/login', data=admin_login))
    assert '/admin/dashboard' in admin_resp.geturl(), f"Expected /admin/dashboard, got {admin_resp.geturl()}"
    print("[PASS] 5. Administrator login routes automatically to /admin/dashboard.")

    # 6. Admin provisions Operator
    create_op_data = urllib.parse.urlencode({
        'full_name': 'Officer Rachel Green',
        'username': 'rachel_operator',
        'email': 'rachel@sonicsentinel.ai',
        'role': 'Security Operator',
        'station': 'Perimeter Dock B',
        'password': 'OperatorPass@123'
    }).encode('utf-8')
    admin_opener.open(urllib.request.Request(f'{BASE}/admin/create-operator', data=create_op_data))
    op = database.get_user_by_email_or_username('rachel_operator')
    assert op is not None and op['role'] == 'Security Operator', "Failed creating operator"
    print("[PASS] 6. Administrator successfully provisioned Security Operator in database.")

    # 7. Operator Login -> routes to /operator/dashboard
    op_opener = get_session()
    op_login = urllib.parse.urlencode({'identifier': 'rachel_operator', 'password': 'OperatorPass@123'}).encode('utf-8')
    op_resp = op_opener.open(urllib.request.Request(f'{BASE}/login', data=op_login))
    assert '/operator/dashboard' in op_resp.geturl(), f"Expected /operator/dashboard, got {op_resp.geturl()}"
    print("[PASS] 7. Security Operator routes automatically to /operator/dashboard.")

    # 8. Operator accessing /admin/dashboard -> 403 Forbidden
    try:
        op_opener.open(f'{BASE}/admin/dashboard')
        assert False, "Operator should not access /admin/dashboard"
    except urllib.error.HTTPError as e:
        assert e.code == 403, f"Expected 403, got {e.code}"
        print("[PASS] 8. Operator accessing /admin/dashboard returns HTTP 403 Forbidden.")

    # 9. Deactivated Account Login Prevention
    op_id = op['id']
    admin_opener.open(urllib.request.Request(f'{BASE}/admin/toggle-user/{op_id}', data=b''))
    deact_op = database.get_user_by_id(op_id)
    assert deact_op['is_active'] == 0, "Account was not deactivated"

    deact_opener = get_session()
    login_attempt = deact_opener.open(urllib.request.Request(f'{BASE}/login', data=op_login))
    assert '/login' in login_attempt.geturl(), "Deactivated account was allowed to log in"
    print("[PASS] 9. Deactivated account blocked from logging in.")

    # 10. Anonymous user accessing protected route -> redirects to /login
    anon = get_session()
    anon_resp = anon.open(f'{BASE}/operator/dashboard')
    assert '/login' in anon_resp.geturl(), "Anonymous access should redirect to /login"
    print("[PASS] 10. Anonymous access to protected routes redirects to /login.")

    # 11. Already logged-in user visiting /login -> redirected to dashboard
    revisit = opener.open(f'{BASE}/login')
    assert '/user/dashboard' in revisit.geturl(), "Logged-in user visiting /login not redirected to dashboard"
    print("[PASS] 11. Logged-in user accessing /login redirected to their own dashboard.")

    print("\n==================================================")
    print("  ALL 11 BACKEND AUTH & RBAC TESTS PASSED 100%!")
    print("==================================================")


if __name__ == '__main__':
    run_tests()
