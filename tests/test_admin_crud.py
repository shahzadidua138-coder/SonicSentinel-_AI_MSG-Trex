import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import database, app

app.app.testing = True
client = app.app.test_client()

# Clean up jack_connor if exists
conn = database.get_db_connection()
conn.execute("DELETE FROM users WHERE username = 'jack_connor'")
conn.commit()
conn.close()

# Login as admin
with client.session_transaction() as sess:
    sess['user_id'] = 1
    sess['role'] = 'Administrator'
    sess['username'] = 'admin'

# 1. Admin provisions a new operator (Security Operator)
res = client.post('/admin/create-operator', data={
    'full_name': 'Officer Jack Connor',
    'username': 'jack_connor',
    'email': 'jack@sonicsentinel.ai',
    'role': 'Security Operator',
    'station': 'Perimeter Alpha',
    'password': 'Password@123'
}, follow_redirects=True)
assert res.status_code == 200
u = database.get_user_by_email_or_username('jack_connor')
assert u is not None and u['role'] == 'Security Operator'
uid = u['id']
print('[PASS] Admin create user with Security Operator role')

# 2. Admin edits user
res = client.post(f'/admin/users/{uid}/edit', data={
    'full_name': 'Officer Jack Connor Updated',
    'email': 'jack_updated@sonicsentinel.ai',
    'role': 'Audio Reviewer',
    'station': 'Forensics Bay 2',
    'organization': 'Acoustic Defense',
    'password': ''
}, follow_redirects=True)
assert res.status_code == 200
u_updated = database.get_user_by_id(uid)
assert u_updated['full_name'] == 'Officer Jack Connor Updated'
assert u_updated['role'] == 'Audio Reviewer'
print('[PASS] Admin edit user details and change role to Audio Reviewer')

# 3. Admin quick role change to Maintenance Operator
res = client.post(f'/admin/users/{uid}/change-role', data={'role': 'Maintenance Operator'}, follow_redirects=True)
assert res.status_code == 200
u_maint = database.get_user_by_id(uid)
assert u_maint['role'] == 'Maintenance Operator'
print('[PASS] Admin quick role change to Maintenance Operator')

# 4. Attempt to change role to Administrator -> blocked!
res = client.post(f'/admin/users/{uid}/change-role', data={'role': 'Administrator'}, follow_redirects=True)
u_after = database.get_user_by_id(uid)
assert u_after['role'] == 'Maintenance Operator'
print('[PASS] Admin attempt to change user role to Administrator is blocked')

# 5. Admin suspends user
res = client.post(f'/admin/toggle-user/{uid}', follow_redirects=True)
u_susp = database.get_user_by_id(uid)
assert u_susp['is_active'] == 0
print('[PASS] Admin suspends user')

# 6. Suspended user login attempt fails
with client.session_transaction() as sess:
    sess.clear()
res = client.post('/login', data={'identifier': 'jack_connor', 'password': 'Password@123'}, follow_redirects=True)
assert b'deactivated or suspended' in res.data
print('[PASS] Suspended user login is blocked')

# 7. Admin deletes user
with client.session_transaction() as sess:
    sess['user_id'] = 1
    sess['role'] = 'Administrator'
res = client.post(f'/admin/users/{uid}/delete', follow_redirects=True)
assert database.get_user_by_id(uid) is None
print('[PASS] Admin deletes user')

# 8. Admin cannot delete built-in admin
res = client.post('/admin/users/1/delete', follow_redirects=True)
assert database.get_user_by_id(1) is not None
print('[PASS] Admin cannot delete built-in admin account')

print('ALL ADMIN CRUD ENDPOINTS VERIFIED & WORKING PERFECTLY!')
