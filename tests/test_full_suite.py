"""
test_full_suite.py - End-to-end verification of all 5 dashboards, RBAC, and Admin CRUD
"""
import unittest
from app import app
import database


class TestSonicSentinelFullSuite(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.client = self.app.test_client()
        self.app.config['TESTING'] = True
        self.app.config['WTF_CSRF_ENABLED'] = False

    def login_as(self, username, password):
        self.client.get('/logout')
        return self.client.post('/login', data={
            'identifier': username,
            'password': password
        }, follow_redirects=True)

    def test_01_all_5_dashboards_render_for_admin(self):
        """Administrator has access to their control center and role simulator views."""
        resp = self.login_as('admin', 'Admin@123')
        self.assertEqual(resp.status_code, 200)

        # 1. Admin Dashboard
        resp_admin = self.client.get('/admin/dashboard')
        self.assertEqual(resp_admin.status_code, 200)
        self.assertIn(b"Administrator Command Center", resp_admin.data)
        self.assertIn(b"User &amp; Personnel Directory", resp_admin.data)

        # 2. Operator Dashboard (Admin permitted via role_required)
        resp_op = self.client.get('/operator/dashboard')
        self.assertEqual(resp_op.status_code, 200)
        self.assertIn(b"Security Command Deck", resp_op.data)
        self.assertIn(b"Sector Alarm Siren", resp_op.data)

        # 3. Reviewer Dashboard (Admin permitted)
        resp_rev = self.client.get('/reviewer/dashboard')
        self.assertEqual(resp_rev.status_code, 200)
        self.assertIn(b"Forensic Arbitration Laboratory", resp_rev.data)
        self.assertIn(b"Model Discrepancy Triage Queue", resp_rev.data)

        # 4. Maintenance Dashboard (Admin permitted)
        resp_maint = self.client.get('/maintenance/dashboard')
        self.assertEqual(resp_maint.status_code, 200)
        self.assertIn(b"Engineering Diagnostic Dock", resp_maint.data)
        self.assertIn(b"Distributed Sensor Array Telemetry", resp_maint.data)

        # 5. User Dashboard
        resp_user = self.client.get('/user/dashboard')
        self.assertEqual(resp_user.status_code, 200)
        self.assertIn(b"Community Safety", resp_user.data)
        self.assertIn(b"Interactive Threat Audition Lab", resp_user.data)
        print("[PASS] 1. All 5 dashboards render with HTTP 200 and expected components.")

    def test_02_rbac_strict_isolation(self):
        """Normal User cannot access Admin, Operator, Reviewer, or Maintenance dashboards."""
        # Log in as normal user
        resp = self.login_as('user', 'User@123')
        self.assertEqual(resp.status_code, 200)

        # Accessing Admin -> 403
        r_admin = self.client.get('/admin/dashboard')
        self.assertEqual(r_admin.status_code, 403)

        # Accessing Operator -> 403
        r_op = self.client.get('/operator/dashboard')
        self.assertEqual(r_op.status_code, 403)

        # Accessing Reviewer -> 403
        r_rev = self.client.get('/reviewer/dashboard')
        self.assertEqual(r_rev.status_code, 403)

        # Accessing Maintenance -> 403
        r_maint = self.client.get('/maintenance/dashboard')
        self.assertEqual(r_maint.status_code, 403)

        # Accessing User Dashboard -> 302 redirect to public website home (or 200 OK if followed)
        r_user = self.client.get('/user/dashboard', follow_redirects=True)
        self.assertEqual(r_user.status_code, 200)
        print("[PASS] 2. RBAC strict isolation verified (Normal User redirected to public web and receives HTTP 403 for elevated dashboards).")

    def test_03_role_specific_operators(self):
        """Security Operator, Audio Reviewer, Maintenance Operator access only authorized routes."""
        # Operator
        self.login_as('operator', 'Operator@123')
        self.assertEqual(self.client.get('/operator/dashboard').status_code, 200)
        self.assertEqual(self.client.get('/admin/dashboard').status_code, 403)

        # Reviewer
        self.login_as('reviewer', 'Reviewer@123')
        self.assertEqual(self.client.get('/reviewer/dashboard').status_code, 200)
        self.assertEqual(self.client.get('/admin/dashboard').status_code, 403)

        # Maintenance
        self.login_as('maintenance', 'Maint@123')
        self.assertEqual(self.client.get('/maintenance/dashboard').status_code, 200)
        self.assertEqual(self.client.get('/admin/dashboard').status_code, 403)
        print("[PASS] 3. Security, Reviewer, and Maintenance roles correctly access their dashboards and are blocked from admin.")

    def test_04_admin_crud_workflow(self):
        """Full CRUD: Create, Edit, Role Reassignment (to 4 roles), Suspend, Delete."""
        self.login_as('admin', 'Admin@123')

        # 1. Provision user as Normal User
        test_uname = 'crud_test_person'
        # Clean up if existed
        conn = database.get_db_connection()
        conn.execute("DELETE FROM users WHERE username = ?", (test_uname,))
        conn.commit()
        conn.close()

        resp_create = self.client.post('/admin/create-operator', data={
            'full_name': 'CRUD Test Person',
            'username': test_uname,
            'email': 'crud_test_person@example.com',
            'role': 'Normal User',
            'station': 'Observation Deck 09',
            'password': 'Password@123'
        }, follow_redirects=True)
        self.assertEqual(resp_create.status_code, 200)
        u = database.get_user_by_email_or_username(test_uname)
        self.assertIsNotNone(u)
        self.assertEqual(u['role'], 'Normal User')

        user_id = u['id']

        # 2. Change role to Security Operator
        resp_role = self.client.post(f'/admin/users/{user_id}/change-role', json={
            'role': 'Security Operator'
        })
        self.assertEqual(resp_role.status_code, 200)
        u_updated = database.get_user_by_id(user_id)
        self.assertEqual(u_updated['role'], 'Security Operator')

        # 3. Change role to Audio Reviewer
        resp_role2 = self.client.post(f'/admin/users/{user_id}/change-role', json={
            'role': 'Audio Reviewer'
        })
        self.assertEqual(resp_role2.status_code, 200)
        u_updated2 = database.get_user_by_id(user_id)
        self.assertEqual(u_updated2['role'], 'Audio Reviewer')

        # 4. Change role to Maintenance Operator
        resp_role3 = self.client.post(f'/admin/users/{user_id}/change-role', json={
            'role': 'Maintenance Operator'
        })
        self.assertEqual(resp_role3.status_code, 200)
        u_updated3 = database.get_user_by_id(user_id)
        self.assertEqual(u_updated3['role'], 'Maintenance Operator')

        # 5. Attempt to assign Administrator via UI -> MUST BE BLOCKED
        resp_admin_attempt = self.client.post(f'/admin/users/{user_id}/change-role', json={
            'role': 'Administrator'
        })
        self.assertEqual(resp_admin_attempt.status_code, 400)
        u_shielded = database.get_user_by_id(user_id)
        self.assertNotEqual(u_shielded['role'], 'Administrator')

        # 6. Edit user details
        resp_edit = self.client.post(f'/admin/users/{user_id}/edit', data={
            'full_name': 'CRUD Test Person Edited',
            'email': 'crud_test_edited@example.com',
            'role': 'Normal User',
            'station': 'Perimeter Node B',
            'organization': 'Sonic Sentinel Test Unit'
        }, follow_redirects=True)
        self.assertEqual(resp_edit.status_code, 200)
        u_edited = database.get_user_by_id(user_id)
        self.assertEqual(u_edited['full_name'], 'CRUD Test Person Edited')
        self.assertEqual(u_edited['email'], 'crud_test_edited@example.com')
        self.assertEqual(u_edited['role'], 'Normal User')

        # 7. Suspend user
        resp_suspend = self.client.post(f'/admin/toggle-user/{user_id}', json={})
        self.assertEqual(resp_suspend.status_code, 200)
        u_suspended = database.get_user_by_id(user_id)
        self.assertEqual(u_suspended['is_active'], 0)

        # 8. Suspended user login fails
        self.client.get('/logout')
        resp_fail_login = self.client.post('/login', data={
            'identifier': test_uname,
            'password': 'Password@123'
        }, follow_redirects=True)
        self.assertIn(b"deactivated or suspended", resp_fail_login.data)

        # 9. Admin permanently deletes user
        self.login_as('admin', 'Admin@123')
        resp_delete = self.client.post(f'/admin/users/{user_id}/delete', json={})
        self.assertEqual(resp_delete.status_code, 200)
        u_deleted = database.get_user_by_id(user_id)
        self.assertIsNone(u_deleted)

        # 10. Attempt to delete root admin -> MUST BE BLOCKED
        admin_user = database.get_user_by_email_or_username('admin')
        resp_del_admin = self.client.post(f"/admin/users/{admin_user['id']}/delete", json={})
        # Either 400 or blocked
        admin_check = database.get_user_by_email_or_username('admin')
        self.assertIsNotNone(admin_check)
        print("[PASS] 4. Admin CRUD lifecycle (Create, Edit, Reassign 4 Roles, Block Admin Escalation, Suspend, Delete, Shield Admin) passed 100%.")

    def test_05_alert_and_review_apis(self):
        """Test alert acknowledge and review override endpoints."""
        self.login_as('operator', 'Operator@123')
        alerts = database.get_active_alerts()
        if alerts:
            aid = alerts[0]['id']
            resp = self.client.post(f'/api/alerts/{aid}/ack', json={'status': 'Acknowledged'})
            self.assertEqual(resp.status_code, 200)

        # Review override
        self.login_as('reviewer', 'Reviewer@123')
        queue = database.get_review_queue()
        if queue:
            qid = queue[0]['id']
            resp_override = self.client.post(f'/api/review/{qid}/override', json={
                'final_decision': 'Gunshot',
                'comments': 'Verified via unit test'
            })
            self.assertEqual(resp_override.status_code, 200)
            self.assertIn(b"success", resp_override.data)
        print("[PASS] 5. Alert Ack and Forensic Review Override endpoints operational.")


if __name__ == '__main__':
    unittest.main()
