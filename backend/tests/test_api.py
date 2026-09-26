"""
SonicSentinel AI - API Integration Tests
"""
import pytest
import io
import json
import numpy as np
import soundfile as sf
from pathlib import Path
import sys

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def auth_headers(client):
    """Register and login a test user to obtain JWT authorization headers."""
    rand_id = np.random.randint(10000, 99999)
    email = f"analyst_{rand_id}@sonicsentinel.ai"
    username = f"analyst_{rand_id}"
    password = "SecurePassword123!"

    # 1. Register
    reg_res = client.post("/api/auth/register", json={
        "username": username,
        "email": email,
        "password": password,
        "full_name": "Senior Audio Analyst",
        "role": "audio_reviewer"
    })
    assert reg_res.status_code == 201

    # 2. Login
    login_res = client.post("/api/auth/login", json={
        "email": email,
        "password": password
    })
    assert login_res.status_code == 200
    token = login_res.get_json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def synthetic_wav_bytes():
    """Generate in-memory WAV file bytes."""
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 500 * t)
    
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return buf


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["service"] == "SonicSentinel AI Backend"


def test_register_and_login(client):
    rand_id = np.random.randint(10000, 99999)
    test_email = f"testuser_{rand_id}@sonicsentinel.ai"
    test_username = f"user_{rand_id}"
    password = "Password123!"
    
    # 1. Register
    reg_res = client.post("/api/auth/register", json={
        "username": test_username,
        "email": test_email,
        "password": password,
        "full_name": "Test Analyst"
    })
    assert reg_res.status_code == 201
    reg_data = reg_res.get_json()
    assert reg_data["message"] == "Registration successful"
    assert "user_id" in reg_data
    
    # 2. Login
    login_res = client.post("/api/auth/login", json={
        "email": test_email,
        "password": password
    })
    assert login_res.status_code == 200
    login_data = login_res.get_json()
    assert "access_token" in login_data
    assert login_data["user"]["email"] == test_email


def test_dashboard_stats(client, auth_headers):
    response = client.get("/api/dashboard/stats", headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert "total_events" in data
    assert "critical_alerts" in data
    assert "category_breakdown" in data


def test_alerts_endpoint(client, auth_headers):
    response = client.get("/api/alerts", headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert "alerts" in data
    assert isinstance(data["alerts"], list)


def test_review_queue(client, auth_headers):
    response = client.get("/api/review/queue", headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert "review_queue" in data
    assert isinstance(data["review_queue"], list)


def test_events_endpoint(client, auth_headers):
    response = client.get("/api/events", headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert "events" in data
    assert "total" in data
